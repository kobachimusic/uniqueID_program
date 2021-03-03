import os

class add_program():

    def __init__(self,na):
        self.na = na
        

    def create_folder(self,na):
        
        if not os.path.exists("./subject/"+self.na+"/analysis_data"):
            os.mkdir("./subject/"+self.na+"/analysis_data")
        if not os.path.exists("./subject/"+self.na+"/anq"):
            os.mkdir("./subject/"+self.na+"/anq")
        if not os.path.exists("./subject/"+self.na+"/thermo"):
            os.mkdir("./subject/"+self.na+"/thermo")

    