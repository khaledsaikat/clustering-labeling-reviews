#!/usr/bin/env python3
import unittest
from context import helpers

#9625993428, 1400599997, 1400501466

class HelpersTest(unittest.TestCase):

    def testLoadJsonToList(self):
        data = helpers.loadJsonToList("data/reviews.json")
        self.assertIsInstance(data, list)
        self.assertTrue(len(data) > 0)


    def testGroupByProducts(self):
        reviews = helpers.loadJsonToList("data/reviews.json")
        data = helpers.groupByProducts(reviews)
        #print(len(data))
        #for k,v in data.items():
        #    print(len(k), len(v), v[0]["asin"])

        self.assertIsInstance(data, dict)
        self.assertTrue(len(data) > 0)
        #self._count_dict_items(data)


    def testGetSingleProductReviews(self):
        reviews = helpers.getSingleProductReviews("9625993428")
        print(len(reviews))
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
        data = [v for v in data.values()]
        for d in data:
            [print(v["reviewText"]) for v in d]

    def testwriteToFile(self):
        import json

        targetFile = open("output.txt", "w");

        reviews = helpers.getSingleProductReviews("9625993428")
        for review in reviews:
            targetFile.write(json.dumps(review))
            targetFile.write("\n")

        reviews = helpers.getSingleProductReviews("1400599997")
        for review in reviews:
            targetFile.write(json.dumps(review))
            targetFile.write("\n")

        reviews = helpers.getSingleProductReviews("1400501466")
        for review in reviews:
            targetFile.write(json.dumps(review))
            targetFile.write("\n")

        targetFile.close()



if __name__ == "__main__":
    unittest.main()
