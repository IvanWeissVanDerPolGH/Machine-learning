class PointObject:
    def __init__(self, name, color,Data_x,Data_y):
        """
        Initialize a PointGroup instance.

        Parameters:
        group_name (str): The name of the group.
        group_color (str): The color of the group.
        Data_x (float): Data_x coordinate of the point.
        Data_y (float): Data_y coordinate of the point.
        """
        self.name = name
        self.color = color
        self.Data_x = Data_x
        self.Data_y = Data_y
        
    def add_point(self, name, color, x, y):
        """
        Add a point to the group.

        Parameters:
        group_name (str): The name of the group.
        group_color (str): The color of the group.
        Data_x (float): Data_x coordinate of the point.
        Data_y (float): Data_y coordinate of the point.
        """
        self.points.append(name, color, x, y)
