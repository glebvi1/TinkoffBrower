from data.search import retrieve, build_index
from math import log2

data_query = {
    "shape of you": (6, 6, 4, 5, 4),
    "beatles": (6, 6, 6, 6, 6),
    "big": (6, 4, 4, 5, 4),
    "little": (6, 6, 6, 6, 5),
    "little big": (6, 5, 4, 4, 4),
    "tea": (6, 6, 6, 6, 5),
}

data_rank = {
    0.0: 1,
    0.2: 2,
    0.4: 3,
    0.6: 4,
    0.8: 5,
    1.0: 6
}


class Metrics:
    def __init__(self, count=10):
        build_index()
        self.data = {}

        for query in data_query.keys():
            base_index = retrieve(query, local_count_candidates=count, return_base_index=True)
            self.data[query] = tuple(map(lambda x: Metrics.get_round_score(x), base_index))

    def precision_k(self, query: str, k: int) -> float:
        if not Metrics.check_query(query):
            return 0
        count_access = sum([1 for access_rank, pred_rank in
                            zip(data_query[query][:k], self.data[query][:k])
                        if access_rank == pred_rank])
        return round(count_access / k, ndigits=2)

    def print_metrics(self):
        for query in data_query:
            print()
            print(query)
            for x in range(1, 6):
                print(f"MRR@{x} = {Metrics.mrr_k(query, k=x)}")
                print(f"Precision@{x} = {self.precision_k(query, k=x)}")
            print(f"DCG = {Metrics.dcg(query)}")

    @staticmethod
    def mrr_k(query: str, k: int):
        if not Metrics.check_query(query):
            return 0
        mrr = sum([1 / rank for rank in data_query[query][:k]]) / k
        return round(mrr, ndigits=2)

    @staticmethod
    def dcg(query: str):
        if not Metrics.check_query(query):
            return 0
        score = sum([rank / log2(i + 2) for i, rank in enumerate(data_query[query])])
        i_score = sum([rank / log2(i + 2) for i, rank in enumerate(sorted(data_query[query], key=lambda x: -x))])

        return round(score / i_score, ndigits=2)

    @staticmethod
    def get_round_score(scores):
        new_score = 0
        for k, v in data_rank.items():
            if scores <= k:
                new_score = v
                break
        return new_score

    @staticmethod
    def check_query(query: str):
        return True if query in data_query else False


if __name__ == "__main__":
    m = Metrics()
    m.print_metrics()
