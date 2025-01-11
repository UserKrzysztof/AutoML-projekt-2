import json
import os

class ResultsReader():
    def __init__(self, path):
        self.path = path
        self.results = self.load_results()
    
    def load_results(self):
        results_dir = self.path
        results = {}
        for file in os.listdir(results_dir):
            with open(os.path.join(results_dir, file), 'r') as f:
                wrapper_name = os.path.splitext(file)[0].replace('_results', '')
                results[wrapper_name] = json.load(f)
        return results