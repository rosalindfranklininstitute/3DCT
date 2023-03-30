import random
class cLeaveOneOut:
    '''
    Iterator for getting a list of elements with one element out
    '''
    def __init__(self, elem_list, order=None):
        '''
        Paramters
            elem_list
            order: None for sequential order, 'random' for random one-out order
        '''

        if len(elem_list)>0:
            self.elem_list=elem_list
            self.order=order
        else:
            print("cLeaveOneOut: no elements")
            return None


    def __iter__(self):
        # Called when iterator is initialized

        self.index_order = list( range(0,len(self.elem_list)) )

        if self.order=='random':
            random.shuffle(self.index_order) #Shuffle in place
        
        self.iteration = 0

        return self

    def __next__(self):

        if self.iteration >= len(self.elem_list):
            raise StopIteration

        #Copy list
        list_missing_one = self.elem_list.copy()

        excluded_item=list_missing_one.pop( self.index_order[self.iteration] )
        self.iteration+=1

        return list_missing_one, excluded_item