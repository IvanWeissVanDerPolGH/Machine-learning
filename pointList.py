from pointObject import PointObject


class pointList:
    def __init__(self, group_name, group_color, group_points: list[tuple]):
        """
        Initialize a PointGroup instance.

        Parameters:
        group_name (str): The name of the group.
        group_color (str): The color of the group.
        """
        self.group_points = []
        
        for point in group_points:
            new_point = PointObject(name= group_name, color= group_color, points= point)
            self.group_points.append(new_point)
            