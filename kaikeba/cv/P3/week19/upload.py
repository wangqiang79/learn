from modelarts.session import Session
import sys


if __name__=="__main__":
	session = Session()
       
	#session.upload_data(bucket_path=sys.argv[1], path=sys.argv[2])
	session.upload_data(bucket_path="/0319-openclass/0428shiyan/", path=sys.argv[1])
