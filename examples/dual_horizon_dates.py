from datetime import date, timedelta


def get_data_dates(current_date: date | None = None) -> tuple[str, str]:
    current_date = current_date or date.today()
    data_date = current_date
    if current_date.weekday() == 5:
        data_date = current_date - timedelta(days=1)
    elif current_date.weekday() == 6:
        data_date = current_date - timedelta(days=2)

    previous_date = data_date - timedelta(days=1)
    return data_date.strftime("%Y%m%d"), previous_date.strftime("%Y%m%d")
