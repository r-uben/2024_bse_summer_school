import os


class Paths:


    def __init__(self):

        self.__homework = None
        self.__results = None
        self.__data = None
        self.__fig = None

    @property
    def homework(self):
        if self.__homework is None:
            self.__homework = os.path.dirname(os.path.dirname(__file__))
        return self.__homework
        
        
    @property
    def results(self):
        if self.__results is None:
            self.__results = os.path.join(self.homework, "results")
        return self.__results
    
    @property
    def data(self):
        if self.__data is None:
            self.__data = os.path.join(self.results, "data")
        return self.__data
    

    @property
    def fig(self):
        if self.__fig is None:
            self.__fig = os.path.join(self.results, "fig")
        return self.__fig

    