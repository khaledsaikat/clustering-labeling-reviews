#!/usr/bin/env python3
import helpers
import data_loader as dl

def run():
    reviews = dl.loadJsonByAspectBraces("../data/kindle501.json")
    aspects = helpers.getTopAspectGroups(reviews)
    return aspects


if __name__ == "__main__":
    print(run())
