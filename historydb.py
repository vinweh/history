import sqlite3
from contextlib import closing

class HistoryDb:
    """Encapsulate the browser's history database"""
    def __init__(self, history_db) -> None:
        self.history_db = history_db

    def get_urls(self, limit=1000):
        """Load the history database rows"""
        with closing(sqlite3.connect(self.history_db)) as connection:
            with closing(connection.cursor()) as c:
                statement = "SELECT url, title FROM urls ORDER BY visit_count DESC LIMIT %(limit)s" % ({"limit" : limit})
                self.rows = c.execute(statement).fetchall()