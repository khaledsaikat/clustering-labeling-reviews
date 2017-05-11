#!/usr/bin/env python3
import unittest
from context import helpers

class HelpersTest(unittest.TestCase):

    def testLoadJsonToList(self):
        data = helpers.loadJsonToList("data/reviews.json")
        self.assertIsInstance(data, list)
        self.assertTrue(len(data) > 0)


    def testGroupByProducts(self):
        reviews = helpers.loadJsonToList("data/reviews.json")
        data = helpers.groupByProducts(reviews)
        self.assertIsInstance(data, dict)
        self.assertTrue(len(data) > 0)
        #self._count_dict_items(data)


    def testGetSingleProductReviews(self):
        reviews = helpers.getSingleProductReviews("0528881469")
        self.assertIsInstance(reviews, list)
        self.assertTrue(len(reviews) > 0)
        #[print(review) for review in reviews]


    def _count_dict_items(self, data):
        """Show number of items for each values in dictionary"""
        for k, v in data.items():
            print(len(v))


    def _testTryOut(self):
        reviews = helpers.loadJsonToList("data/reviews.json")
        data = helpers.groupByProducts(reviews)
        data = [v for v in data.values()][1]
        [print(v) for v in data]


if __name__ == "__main__":
    unittest.main()
