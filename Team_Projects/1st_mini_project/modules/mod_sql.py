import pymysql

class Database():
    def __init__(self) :
        self.db = pymysql.connect(
                                    user = "root",
                                    passwd = " ",
                                    host = "localhost",
                                    db = " "
                                )

        self.cursor = self.db.cursor(pymysql.cursors.DictCursor)

    def execute(self, sql, values={}) :
        self.cursor.execute(sql, values)

    def executeAll(self, sql, values={}) :
        self.cursor.execute(sql, values)
        self.result = self.cursor.fetchall()
        return self.result

    def commit(self) :
        return self.db.commit()
