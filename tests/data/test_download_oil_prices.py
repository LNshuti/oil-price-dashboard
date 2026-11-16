from io import BytesIO
from unittest.mock import MagicMock, patch

import pandas as pd

from scr.data.download_oil_prices import download_and_tidy_oil_prices

URL = "https://www.eia.gov/petroleum/gasdiesel/xls/pswrgvwall.xls"
SUFFIX = (" All Grades All Formulations Retail Gasoline Prices"
          "  (Dollars per Gallon)")


def _eia_workbook() -> bytes:
    """A minimal workbook shaped like EIA's 'Data 12' sheet."""
    rows = [
        ["Back to Contents", "Data 12", None],
        ["Sourcekey", "EMM_EPM0_PTE_NUS_DPG", "EMM_EPM0_PTE_R1X_DPG"],
        ["Date", f"Weekly U.S.{SUFFIX}",
         f"Weekly New England (PADD 1A){SUFFIX}"],
        [pd.Timestamp("2026-11-02"), 4.15, 4.14],
        [pd.Timestamp("2026-11-09"), 4.00, 4.06],
    ]
    buffer = BytesIO()
    pd.DataFrame(rows).to_excel(buffer, sheet_name="Data 12", header=False,
                                index=False)
    return buffer.getvalue()


def test_download_and_tidy_oil_prices(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    response = MagicMock(content=_eia_workbook())

    with patch("scr.data.download_oil_prices.requests.get",
               return_value=response) as get:
        df = download_and_tidy_oil_prices(URL)

    get.assert_called_once_with(URL)
    assert list(df.columns) == ["Date", "U.S.", "New England"]
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    assert df["U.S."].tolist() == [4.15, 4.00]
    saved = tmp_path / "data" / "processed" / "weekly_gas_price_data.csv"
    assert saved.exists()
