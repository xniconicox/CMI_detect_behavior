"""Lower level implementation details of the gateway.
Hosts should not need to review this file before writing their competition specific gateway.
"""

import enum
import json
import os
import pathlib
import re
import subprocess

from socket import gaierror
from typing import Any, List, Optional, Tuple, Union

import grpc
import numpy as np
import pandas as pd
import polars as pl

import kaggle_evaluation.core.relay


# Files in this directory are visible to the competitor container.
_FILE_SHARE_DIR = '/kaggle/shared/'
IS_RERUN = os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None


class GatewayRuntimeErrorType(enum.Enum):
    """Allow-listed error types that Gateways can raise, which map to canned error messages to show users.
    Please try capture all errors with one of these types.
    Unhandled errors are treated as caused by Kaggle and do not count towards daily submission limits.
    """

    UNSPECIFIED = 0
    SERVER_NEVER_STARTED = 1
    SERVER_CONNECTION_FAILED = 2
    SERVER_RAISED_EXCEPTION = 3
    SERVER_MISSING_ENDPOINT = 4
    # Default error type if an exception was raised that was not explicitly handled by the Gateway
    GATEWAY_RAISED_EXCEPTION = 5
    INVALID_SUBMISSION = 6
    GRPC_DEADLINE_EXCEEDED = 7


class GatewayRuntimeError(Exception):
    """Gateways can raise this error to capture a user-visible error enum from above and host-visible error details."""

    def __init__(self, error_type: GatewayRuntimeErrorType, error_details: Optional[str] = None):
        self.error_type = error_type
        self.error_details = error_details


class BaseGateway:
    def __init__(self, data_paths: Tuple[str] = None, file_share_dir: Optional[str] = _FILE_SHARE_DIR, target_column_name: Optional[str] = None):
        self.client = kaggle_evaluation.core.relay.Client('inference_server' if IS_RERUN else 'localhost')
        self.server = None  # The gateway can have a server but it isn't typically necessary.
        # Off Kaggle, we can accept a user input file_share_dir. On Kaggle, we need to use the special directory
        # that is visible to the user.
        if file_share_dir or not os.path.exists('/kaggle'):
            self.file_share_dir = file_share_dir
        else:
            self.file_share_dir = _FILE_SHARE_DIR

        self._shared_a_file = False
        self.data_paths = data_paths
        self.target_column_name = (
            target_column_name  # Only used if the predictions are made as a primitive type (int, bool, etc) rather than a dataframe.
        )

    def validate_prediction_batch(self, prediction_batch: Any, row_ids: Union[pl.DataFrame, pl.Series, pd.DataFrame, pd.Series]) -> None:
        """If competitors can submit fewer rows than expected they can save all predictions for the last batch and
        bypass the benefits of the Kaggle evaluation service. This attack was seen in a real competition with the older time series API:
        https://www.kaggle.com/competitions/riiid-test-answer-prediction/discussion/196066
        It's critically important that this check be run every time predict() is called.

        If your predictions may take a variable number of rows and you need to write a custom version of this check,
        you still must specify a minimum row count greater than zero per prediction batch.
        """
        if prediction_batch is None:
            raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, 'No prediction received')
        num_received_rows = None
        # Special handling for numpy ints only as numpy floats are python floats, but numpy ints aren't python ints
        for primitive_type in [int, float, str, bool, np.int_]:
            if isinstance(prediction_batch, primitive_type):
                # Types that only support one predictions per batch don't need to be validated.
                # Basic types are valid for prediction, but either don't have a length (int) or the length isn't relevant for
                # purposes of this check (str).
                num_received_rows = 1
        if num_received_rows is None:
            if type(prediction_batch) not in [pl.DataFrame, pl.Series, pd.DataFrame, pd.Series]:
                raise GatewayRuntimeError(
                    GatewayRuntimeErrorType.INVALID_SUBMISSION, f'Invalid prediction data type, received: {type(prediction_batch)}'
                )
            num_received_rows = len(prediction_batch)
        if type(row_ids) not in [pl.DataFrame, pl.Series, pd.DataFrame, pd.Series]:
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, f'Invalid row ID type {type(row_ids)}; expected Polars DataFrame or similar'
            )
        num_expected_rows = len(row_ids)
        if len(row_ids) == 0:
            raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, 'Missing row IDs for batch')
        if num_received_rows != num_expected_rows:
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.INVALID_SUBMISSION, f'Invalid predictions: expected {num_expected_rows} rows but received {num_received_rows}'
            )

    def _standardize_and_validate_paths(self, input_paths: List[Union[str, pathlib.Path]]) -> Tuple[List[str], List[str]]:
        # Accept a list of str or pathlib.Path, but standardize on list of str
        if input_paths and not self.file_share_dir or type(self.file_share_dir) not in (str, pathlib.Path):
            raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, f'Invalid `file_share_dir`: {self.file_share_dir}')

        for path in input_paths:
            if os.pardir in str(path):
                raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, f'Send files path contains {os.pardir}: {path}')
            if str(path) != str(os.path.normpath(path)):
                # Raise an error rather than sending users unexpectedly altered paths
                raise GatewayRuntimeError(
                    GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, f'Send files path {path} must be normalized. See `os.path.normpath`'
                )
            if type(path) not in (pathlib.Path, str):
                raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, 'All paths must be of type str or pathlib.Path')
            if not os.path.exists(path):
                raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, f'Input path {path} does not exist')

        input_paths = [os.path.abspath(path) for path in input_paths]
        if len(set(input_paths)) != len(input_paths):
            raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, 'Duplicate input paths found')

        output_dir = str(self.file_share_dir)
        if not output_dir.endswith(os.path.sep):
            # Ensure output dir is valid for later use
            output_dir += os.path.sep

        # Can't use os.path.join for output_dir + path: os.path.join won't prepend to an abspath
        # normpath manages // in particular.
        output_paths = [os.path.normpath(output_dir + path) for path in input_paths]
        return input_paths, output_paths

    def share_files(
        self,
        input_paths: List[Union[str, pathlib.Path]],
    ) -> List[str]:
        """Makes files and/or directories available to the user's inference_server. They will be mirrored under the
        self.file_share_dir directory, using the full absolute path. An input like:
            /kaggle/input/mycomp/test.csv
        Would be written to:
            /kaggle/shared/kaggle/input/mycomp/test.csv

        Args:
            input_paths: List of paths to files and/or directories that should be shared.

        Returns:
            The output paths that were shared.

        Raises:
            GatewayRuntimeError if any invalid paths are passed.
        """
        if self.file_share_dir and not self._shared_a_file:
            if os.path.exists(self.file_share_dir) and (not os.path.isdir(self.file_share_dir) or len(os.listdir(self.file_share_dir)) > 0):
                raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, '`file_share_dir` must be an empty directory.')
            os.makedirs(self.file_share_dir, exist_ok=True)
            self._shared_a_file = True

        if not input_paths:
            raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, 'share_files requires at least one input path')

        input_paths, output_paths = self._standardize_and_validate_paths(input_paths)
        for in_path, out_path in zip(input_paths, output_paths):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # This makes the files available to the InferenceServer as read-only. Only the Gateway can mount files.
            # mount will only work in live kaggle evaluation rerun sessions. Otherwise use a symlink.
            if IS_RERUN:
                if not os.path.isdir(out_path):
                    pathlib.Path(out_path).touch()
                try:
                    subprocess.run(f'mount --bind {in_path} {out_path}', shell=True, check=True)
                except Exception:
                    # `mount`` is expected to be faster but less reliable in our context.
                    # Fall back to cp if possible.
                    if self.file_share_dir != _FILE_SHARE_DIR:
                        raise GatewayRuntimeError(
                            GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION,
                            f'share_files fallback failure: can only use cp if file_share_dir is {_FILE_SHARE_DIR}. Got {self.file_share_dir}',
                        )
                    # cp will fail if the output directory doesn't already exist
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    subprocess.run(f'cp -r {in_path} {out_path}', shell=True, check=True)
            else:
                subprocess.run(f'ln -s {in_path} {out_path}', shell=True, check=True)

        return output_paths

    def write_submission(self, predictions, row_ids: List[Union[pl.Series, pl.DataFrame, pd.Series, pd.DataFrame]]) -> None:
        """Export the predictions to a submission.parquet."""
        if isinstance(predictions, list):
            if isinstance(predictions[0], pd.DataFrame):
                predictions = pd.concat(predictions, ignore_index=True)
            elif isinstance(predictions[0], pl.DataFrame):
                try:
                    predictions = pl.concat(predictions, how='vertical_relaxed')
                except pl.exceptions.SchemaError:
                    raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, 'Inconsistent prediction types')
                except pl.exceptions.ComputeError:
                    raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, 'Inconsistent prediction column counts')
            else:
                if type(row_ids[0]) in [pl.Series, pl.DataFrame]:
                    row_ids = pl.concat(row_ids)
                elif type(row_ids[0]) in [pd.Series, pd.DataFrame]:
                    row_ids = pd.concat(row_ids).reset_index(drop=True)
                else:
                    raise GatewayRuntimeError(
                        GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION,
                        f'Invalid row ID datatype {type(row_ids[0])}. Expected Polars series or dataframe.',
                    )
                if self.target_column_name is None:
                    raise GatewayRuntimeError(
                        GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, '`target_column_name` must be set in order to use scalar value predictions.'
                    )
                predictions = pl.DataFrame(data={row_ids.columns[0]: row_ids, self.target_column_name: predictions})

        if isinstance(predictions, pd.DataFrame):
            predictions.to_parquet('submission.parquet', index=False)
        elif isinstance(predictions, pl.DataFrame):
            pl.DataFrame(predictions).write_parquet('submission.parquet')
        else:
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, f"Unsupported predictions type {type(predictions)}; can't write submission file"
            )

    def write_result(self, error: Optional[GatewayRuntimeError] = None) -> None:
        """Export a result.json containing error details if applicable."""
        result = {'Succeeded': error is None}

        if error is not None:
            result['ErrorType'] = error.error_type.value
            result['ErrorName'] = error.error_type.name
            # Max error detail length is 8000
            result['ErrorDetails'] = str(error.error_details[:8000]) if error.error_details else None

        with open('result.json', 'w') as f_open:
            json.dump(result, f_open)

    def handle_server_error(self, exception: Exception, endpoint: str) -> None:
        """Determine how to handle an exception raised when calling the inference server. Typically just format the
        error into a GatewayRuntimeError and raise.
        """
        exception_str = str(exception)
        if isinstance(exception, gaierror) or (isinstance(exception, RuntimeError) and 'Failed to connect to server after waiting' in exception_str):
            raise GatewayRuntimeError(GatewayRuntimeErrorType.SERVER_NEVER_STARTED) from None
        if f'No listener for {endpoint} was registered' in exception_str:
            raise GatewayRuntimeError(GatewayRuntimeErrorType.SERVER_MISSING_ENDPOINT, f'Server did not register a listener for {endpoint}') from None
        if 'Exception calling application' in exception_str:
            # Extract just the exception message raised by the inference server
            message_match = re.search('"Exception calling application: (.*)"', exception_str, re.IGNORECASE)
            message = message_match.group(1) if message_match else exception_str
            raise GatewayRuntimeError(GatewayRuntimeErrorType.SERVER_RAISED_EXCEPTION, message) from None
        if isinstance(exception, grpc._channel._InactiveRpcError):
            raise GatewayRuntimeError(GatewayRuntimeErrorType.SERVER_CONNECTION_FAILED, exception_str) from None
        if isinstance(exception, kaggle_evaluation.core.relay.GRPCDeadlineError):
            raise GatewayRuntimeError(GatewayRuntimeErrorType.GRPC_DEADLINE_EXCEEDED, exception_str) from None

        raise exception
