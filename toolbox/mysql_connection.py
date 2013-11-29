import MySQLdb
from dbinfo import *

dbcase = {'name_db':name_db}

def sql(fun):
	def circle(dn,sql):
		dbname = fun(dn,sql)['dbname']
		host, port, user, passwd, db = dbcase[dbname]['host'], dbcase[dbname]['port'], dbcase[dbname]['user'], dbcase[dbname]['passwd'], dbcase[dbname]['db']
		Connection = MySQLdb.connect(host=host, port=port, user=user,passwd=passwd, db=db)
		dbh = Connection.cursor()
		dbh.execute(fun(dn,sql)['sql'])
		rec = dbh.fetchall()
		dbh.close()
		return rec
	return circle

def conopen(dbname):
	host, port, user, passwd, db = dbcase[dbname]['host'], dbcase[dbname]['port'], dbcase[dbname]['user'], dbcase[dbname]['passwd'], dbcase[dbname]['db']
	Connection = MySQLdb.connect(host=host, port=port, user=user,passwd=passwd, db=db)
	return Connection.cursor() 
		

###usecase

#from toolbox import mysql_connection
#@myconnection.sql
#def get_sql(dbname, sql):
#	return {'dbname':dbname, 'sql':sql}

#for col in get_sql(dbname,sql):...
