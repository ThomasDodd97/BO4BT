class dimension(object):
    """
    A class for the creation of dimsension objects.
    """
    def __init__(self,name,low,high):
        self.name = name
        self.low = low
        self.high = high

class chemical(object):
    """
    This is a class that allows for the generation of
    chemical objects which can be interacted with in the
    main program.
    Glycerol and Organic Acid
        Glycerol : Organic Acid
                G : OA
    1.0 : 0.5 <=========> 1.0 : 2.0
        0.0 <---------------> 1.0
                    x1
    (a =   OAsr)
    OAsr = Organic Acid Stoichiometric Ratio
    You can set the high and low points in terms of
    stoichiometry. The molar mass. And names.
    """
    def __init__(self,low,high,mr,name=None,abbrev_name=None):
        self.name = name
        self.abbrev_name = abbrev_name
        self.low = low
        self.high = high
        self.mr = mr
        self.alpha = high - low

def string_user_input_retriever(question, ans1, ans2):
    """ 
    This function takes a question that should have specific string based commands
    as answers and makes sure that the user has a command worth something before
    returning it to the main script.

    This function takes a number of variables as inputs:
        question = string, a question to be posed to a user
        ans1 = string, a suitable answer by the user that stops the script
        ans2 = string, a suitable answer by the user that continues the script

    This function returns a single variable:
        user_input = string, an acceptable text-based answer to the question is returned
    """
    while 1 == 1:
        print(question)
        user_input = input(question)
        print("\t\tInput: ", user_input)
        if user_input == ans1:
            break
        elif user_input == ans2:
            break
    return user_input

def numeric_user_input_retriever(question):
    """
    This function takes a question that should have some sort of numerical answer
    and poses it to the user. It then looks at a users answer, decides whether it
    is numeric or string based, before either posing the same question again, or
    returning the successfully retrieved numeric value back as a float.

    This function takes a single variable an an input:
        question = string, the question to be posed to a user

    This function returns a single variable:
        user_input = string, an acceptably numeric answer to the question is returned
    """
    print(question)
    while 1 == 1:
        user_input = input(question)
        try:
            user_input = float(user_input)
            break
        except ValueError:
            print('Please enter a number.')
    print("\t\tInput: ", user_input)
    return user_input

class query():
    """
    This class provides some structure for the various methods
    of querying a user.
    """
    def __init__(self):
        self.name = "query"
    def string(self,arg1,arg2,arg3):
        user_input = string_user_input_retriever(arg1,arg2,arg3)
        return user_input
    def numeric(self,arg1):
        user_input = numeric_user_input_retriever(arg1)
        return user_input