import sqlite3
from contextlib import closing

class HistoryDb:
    """Encapsulate the browser's history database"""
    def __init__(self, history_db) -> None:
        self.history_db = history_db
        self.history_db_uri = f"file:{history_db}?mode=ro" # open read-only

    def get_urls(self, limit=1000):
        """Get urls and titles from the history database"""
        rows = []
        try:
            with closing(sqlite3.connect(self.history_db_uri, uri=True)) as connection:
                with closing(connection.cursor()) as c:
                    statement = "SELECT DISTINCT url, title FROM urls ORDER BY visit_count DESC LIMIT %(limit)s" % ({"limit" : limit})
                    rows = c.execute(statement).fetchall()
        except sqlite3.OperationalError as err:
            msg = f"Exception: sqllite3.OperationalError >> {err} <<. Make sure the browser is not running with the selected profile."
            print(f"{msg}")
        return rows
