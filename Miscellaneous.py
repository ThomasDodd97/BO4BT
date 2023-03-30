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