class PointCloud:
    def __init__(self):
        self.points = []
        self.point_id_list = []
        self.point_desc_list = []
        self.point_xyz_list = []

    def xyz_nearest(self, xyz, nb_neighbors=5):
        pass

    def desc_nearest(self, desc, nb_neighbors=2):
        pass


class Point3D:
    def __init__(self, index, descriptor, xyz, rgb):
        self.index = index
        self.desc = descriptor
        self.xyz = xyz
        self.rgb = rgb
        self.visual_word = None

    def match(self, desc):
        pass

    def assign_visual_word(self):
        pass

