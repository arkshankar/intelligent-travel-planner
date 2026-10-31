import unittest

from travel_agent import (
    TravelKnowledgeBase,
    UtilityBasedAgent,
    astar_route,
    bfs_route,
    dfs_route,
    greedy_route,
    make_request,
    uniform_cost_route,
)


class SearchAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.kb = TravelKnowledgeBase()

    def test_all_searches_return_valid_paths(self):
        searches = (bfs_route, dfs_route, uniform_cost_route, greedy_route, astar_route)
        for search in searches:
            with self.subTest(search=search.__name__):
                route = search(self.kb, "Delhi", "Goa")
                self.assertEqual(route.cities[0], "Delhi")
                self.assertEqual(route.cities[-1], "Goa")
                self.assertEqual(len(route.legs), len(route.cities) - 1)

    def test_ucs_and_astar_find_same_low_cost_route(self):
        ucs = uniform_cost_route(self.kb, "Delhi", "Goa")
        astar = astar_route(self.kb, "Delhi", "Goa")
        self.assertEqual(ucs.cost, astar.cost)
        self.assertEqual(ucs.cities, astar.cities)

    def test_bfs_minimizes_number_of_edges(self):
        route = bfs_route(self.kb, "Delhi", "Goa")
        self.assertEqual(len(route.legs), 2)


class AgentTests(unittest.TestCase):
    def test_utility_agent_creates_explainable_itinerary(self):
        request = make_request("Delhi", ["Jaipur", "Agra"], 25000, 6, ["history"], algorithm="astar")
        itinerary = UtilityBasedAgent().plan(request)
        self.assertEqual(itinerary.route.cities[0], "Delhi")
        self.assertIn("Jaipur", itinerary.eligible_destinations)
        self.assertGreater(itinerary.total_cost, 0)
        self.assertTrue(itinerary.reasoning)

    def test_unmatched_interest_is_rejected(self):
        request = make_request("Delhi", ["Jaipur"], 25000, 5, ["space-travel"])
        with self.assertRaises(ValueError):
            UtilityBasedAgent().plan(request)


if __name__ == "__main__":
    unittest.main()

