import pandas as pd
import autopocket.algorithms.base.BaseSearcher as BaseSearcher

class ModelsLeaderboard:

    @staticmethod
    def createLeaderBoard():
        baseSearcher = BaseSearcher()
        res = baseSearcher.read_results()
        leaderboard = pd.DataFrame(res)

