import tools as tls

class cache():
    ## Simple caching class specific to the floodit solvers 
    __max_depth = 17   #educated guess for max memory used in cache
    __cache = {}
    __calls = 0 
    __miss  = 1
    __update = 0


    def print_cache_stats(self):
        print("Cache statistics:\n - miss:    %s \n - updates: %s \n - calls:   %s \n - hitratio: %s" % self.get_stats())

    def get_stats(self):
        return self.__miss, self.__update, self.__calls, (str((self.__calls - self.__miss) * 100.0 / max(1,self.__calls))+ "%")

    def print_cache(self):
        for x in sorted(self.__cache, reverse=False):
            print(self.__cache[x])
            print("---", x)

    def cached(self, t, s):
        self.__calls += 1
        if t > self.__max_depth:
            # limit acces to cache, such that memory footprint remains low.
            self.__miss += 1
            return False

        h = tls.hashOfSet(s)
        if not h in self.__cache:
            self.__cache[h] = t
            self.__miss += 1
            return False
        else:
            if t < self.__cache[h]:
                self.__cache[h] = t
                self.__update += 1
                return False
            else:
                return True

    def clear_cache(self):
        self.__cache = {}
        # for i in self.__cache: del i ??