import warnings
import unittest
import pandas as pd
from src.llm import LLM

# Suppress all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)

class TestLLM(unittest.TestCase):
    def setUp(self):
        self.llm = LLM()

    def test_generate_samples(self):
        num_samples = 5
        classes = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        generated_samples = self.llm.generate_samples(num_samples, classes)
        
        # Check if the generated samples is a list
        self.assertTrue(isinstance(generated_samples, list))
        
        # Check if the length of the generated samples is correct
        self.assertEqual(len(generated_samples), num_samples * len(classes))
        
        # Check if each sample has the expected keys
        for sample in generated_samples:
            self.assertIn("text", sample)
            self.assertIn("label", sample)
            
        # Check if the labels are within the expected classes
        for sample in generated_samples:
            self.assertIn(sample["label"], classes)

if __name__ == "__main__":
    unittest.main()