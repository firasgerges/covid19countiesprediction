class Attribute(object):
	def __init__(self):
		self.name=None;
		self.is_category=0;
		self.sub_attributes=[];#array all 0s. each element will be a new attribute representing wether or not the original attribute is equal to the option on this index.
		self.original_options=[];#categories of the attributes
		self.indices=[];
