import itertools

def Merge(dict1, dict2): 
    res = {**dict1, **dict2} 
    return res 

class DictIter:
    def __init__(self, args):
        self.args = args
        for k, v in self.args.items():   # all the values must be list
            if not isinstance(v, list):
                self.args[k] = [v]
    
    def reorder(self, keys): # topping the k-v pairs by given keys
        if not isinstance(keys, (tuple, list)):
            keys = [keys]
        if isinstance(keys, tuple):
            keys = list(keys)
        keys.reverse() # !

        for key in keys:
            value = self.args[key]
            del self.args[key]
            self.args[key] = value
    
    def __iter__(self):
        len_list = [len(v) if isinstance(v, list) else 1 for v in self.args.values()]
        len_list = [list(range(i)) for i in len_list]
        iter_list = len_list[0]

        for i in range(1, len(len_list)):
            l = []
            for e in itertools.product(iter_list, len_list[i]):
                if isinstance(e[0], list):
                    zz = [n for n in e[0]]
                    zz.append(e[1])
                    l.append(zz)
                else:
                    l.append(list(e))
            iter_list = l
            
        self.iter_list = iter_list
        self.i = 0

        return self

    def __next__(self):
        if self.i == len(self):
            raise StopIteration

        l = self.iter_list[self.i]
        args = {list(self.args.keys())[i]: list(self.args.values())[i][n] for i, n in enumerate(l)}

        self.i += 1
        return args

    def __len__(self):
        return len(self.iter_list)

if __name__ == '__main__':
    # DictIter
    args = {
        'name': ['wang', 'li', 'zhao'],
        'age': [18, 19, 20],
        'loc': ['dalian', 'shenyang', 'beijing'],
        'nation': 'china'
    }
    dict_iter = DictIter(args)
    dict_iter.reorder(['name'])
    args = iter(dict_iter)
    for i, arg in enumerate(args):
        print(i, arg)
    
    # # Merge
    # dict1 = {'a': 10, 'b': 8} 
    # dict2 = {'d': 6, 'c': 4} 
    # dict1 = Merge(dict1, dict2) 
    # print(dict1)