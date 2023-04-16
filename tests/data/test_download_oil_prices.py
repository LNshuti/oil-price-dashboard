import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import io
import re
import requests

#import your_module  # Replace with the actual name of the module containing the above code


class TestGasPriceData(unittest.TestCase):
    @patch("requests.get")
    @patch("lineapy.save")
    def test_gas_price_data_retrieval(self, mock_lineapy_save, mock_requests_get):
        # Prepare a mock response from the EIA website
        mock_excel_data = b"Sample binary data of Excel file"
        mock_response = MagicMock(spec=requests.Response)
        mock_response.content = mock_excel_data
        mock_requests_get.return_value = mock_response

        # Mock the expected DataFrame transformation result
        mock_df = pd.DataFrame({"Date": [], "Price": []})

        # Mock pd.read_excel
        with patch("pd.read_excel", return_value=mock_df) as mock_read_excel:
            your_module.function_containing_the_code()  # Replace with the actual name of the function containing the given code

            # Check if the requests.get was called with the correct URL
            mock_requests_get.assert_called_once_with("https://www.eia.gov/petroleum/gasdiesel/xls/pswrgvwall.xls")

            # Check if pd.read_excel was called with the correct arguments
            mock_read_excel.assert_called_once_with(
                mock_excel_data,
                sheet_name="Data 12",
                index_col=0,
                skiprows=2,
                parse_dates=["Date"],
            )

            # Check if lineapy.save was called with the correct arguments
            mock_lineapy_save.assert_called_once_with(mock_df, "weekly_gas_price_data")


if __name__ == "__main__":
    unittest.main()
