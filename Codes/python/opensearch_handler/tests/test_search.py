from __future__ import annotations

from search import count_documents, hybrid_search, multi_match_search, search_raw


class DummyClient:
    def search(self, **kwargs):
        return kwargs

    def count(self, **kwargs):
        return kwargs


def test_search_builders():
    client = DummyClient()

    raw = search_raw(client, "idx", {"query": {"match_all": {}}})
    assert raw == {"index": "idx", "body": {"query": {"match_all": {}}}}

    mm = multi_match_search(client, "idx", "hello", ["title", "content"], size=3)
    assert mm["body"]["query"]["multi_match"]["fields"] == ["title", "content"]
    assert mm["body"]["size"] == 3

    hybrid = hybrid_search(
        client,
        "idx",
        query="vector",
        text_field="content",
        vector_field="embedding",
        vector=[0.1, 0.2, 0.3],
        k=2,
    )
    should = hybrid["body"]["query"]["bool"]["should"]
    assert should[0] == {"match": {"content": "vector"}}
    assert should[1] == {"knn": {"embedding": {"vector": [0.1, 0.2, 0.3], "k": 2}}}

    count_all = count_documents(client, "idx")
    assert count_all == {"index": "idx", "body": {}}

    count_filtered = count_documents(client, "idx", query={"term": {"kind": "note"}})
    assert count_filtered["body"] == {"query": {"term": {"kind": "note"}}}
