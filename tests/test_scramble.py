import pytest
import random
from unittest.mock import Mock, patch
from reef.obfuscators.scramble import (
    LinearScrambleObfuscator,
    HierarchicalScrambleObfuscator,
    get_nested_dict_from_list,
    deep_update,
    linearise_sentence,
    shuffle_siblings,
    swap_head_directions,
)


class TestHelperFunctions:
    """Test suite for helper functions"""
    
    def test_get_nested_dict_from_list_single_item(self):
        """Test nested dict creation with single item"""
        input_list = [("word", "L")]
        result = get_nested_dict_from_list(input_list)
        expected = {("word", "L"): {}}
        assert result == expected
    
    def test_get_nested_dict_from_list_multiple_items(self):
        """Test nested dict creation with multiple items"""
        input_list = [("the", "L"), ("cat", "L"), ("sat", "R")]
        result = get_nested_dict_from_list(input_list)
        expected = {("the", "L"): {("cat", "L"): {("sat", "R"): {}}}}
        assert result == expected
    
    def test_get_nested_dict_from_list_empty(self):
        """Test nested dict creation with empty list"""
        result = get_nested_dict_from_list([])
        assert result == {}
    
    def test_deep_update_simple(self):
        """Test deep update with simple dicts"""
        main = {"a": 1, "b": 2}
        update = {"b": 3, "c": 4}
        deep_update(main, update)
        assert main == {"a": 1, "b": 3, "c": 4}
    
    def test_deep_update_nested(self):
        """Test deep update with nested dicts"""
        main = {"a": {"x": 1, "y": 2}, "b": 3}
        update = {"a": {"y": 10, "z": 5}}
        deep_update(main, update)
        assert main == {"a": {"x": 1, "y": 10, "z": 5}, "b": 3}
    
    def test_linearise_sentence_simple(self):
        """Test linearising a simple sentence tree"""
        tree = {
            ("root", "R"): {
                ("left", "L"): {},
                ("right", "R"): {}
            }
        }
        result = linearise_sentence(tree)
        assert result == ["left", "root", "right"]
    
    def test_linearise_sentence_reverse(self):
        """Test linearising with reverse flag"""
        tree = {
            ("root", "R"): {
                ("left", "L"): {},
                ("right", "R"): {}
            }
        }
        result = linearise_sentence(tree, reverse=True)
        assert result == ["right", "root", "left"]
    
    def test_linearise_sentence_complex(self):
        """Test linearising a more complex tree"""
        tree = {
            ("main", "R"): {
                ("the", "L"): {},
                ("cat", "L"): {},
                ("sat", "R"): {}
            }
        }
        result = linearise_sentence(tree)
        assert "main" in result
        assert len(result) == 4
    
    def test_shuffle_siblings_maintains_structure(self):
        """Test that shuffle_siblings maintains tree structure"""
        random.seed(42)
        tree = {
            ("root", "R"): {
                ("a", "L"): {},
                ("b", "L"): {},
                ("c", "R"): {},
                ("d", "R"): {}
            }
        }
        result = shuffle_siblings(tree)
        
        # Check that all keys are present
        assert len(result) == 1
        root_children = list(result.values())[0]
        assert len(root_children) == 4
        
        # Check that directions are preserved
        directions = [k[1] for k in root_children.keys()]
        assert directions.count("L") == 2
        assert directions.count("R") == 2
    
    def test_swap_head_directions(self):
        """Test swapping L and R directions"""
        tree = {
            ("root", "R"): {
                ("left", "L"): {},
                ("right", "R"): {}
            }
        }
        result = swap_head_directions(tree)
        
        # Root should swap from R to L
        assert ("root", "L") in result
        children = result[("root", "L")]
        
        # Children should also swap
        child_keys = list(children.keys())
        assert ("left", "R") in child_keys
        assert ("right", "L") in child_keys
    
class TestLinearScrambleObfuscator:
    """Test suite for ScrambleObfuscator class"""
    
    @pytest.fixture
    def obfuscator(self):
        """Create a ScrambleObfuscator instance"""
        return LinearScrambleObfuscator()
    
    def test_linear_scramble_same_seed(self, obfuscator):
        """Test that same seed produces same scramble"""
        text = "the quick brown fox jumps"
        result1 = obfuscator.obfuscate(text, seed=42)
        result2 = obfuscator.obfuscate(text, seed=42)
        assert result1 == result2
    
    def test_linear_scramble_different_seed(self, obfuscator):
        """Test that different seeds produce different scrambles"""
        text = "the quick brown fox jumps"
        result1 = obfuscator.obfuscate(text, seed=42)
        result2 = obfuscator.obfuscate(text, seed=99)
        # Very unlikely to be the same with different seeds
        assert result1 != result2 or len(text.split()) <= 1
    
    def test_linear_scramble_preserves_words(self, obfuscator):
        """Test that linear scramble preserves all words"""
        text = "the quick brown fox jumps"
        result = obfuscator.obfuscate(text, seed=42)
        
        original_words = set(text.split())
        scrambled_words = set(result.split())
        assert original_words == scrambled_words
    
    def test_linear_scramble_single_word(self, obfuscator):
        """Test linear scramble with single word"""
        text = "hello"
        result = obfuscator.obfuscate(text, seed=42)
        assert result == "hello"
    
    def test_linear_scramble_empty_string(self, obfuscator):
        """Test linear scramble with empty string"""
        text = ""
        result = obfuscator.obfuscate(text, seed=42)
        assert result == ""
    
class TestHierarchicalScrambleObfuscator:
    @pytest.fixture
    def obfuscator(self):
        """Create a ScrambleObfuscator instance"""
        return HierarchicalScrambleObfuscator()

    def test_get_route_to_root_single_token(self, obfuscator):
        """Test route to root for a single token (root itself)"""
        mock_token = Mock()
        mock_token.text = "root"
        mock_token.i = 0
        mock_token.head = mock_token  # Points to itself (is root)
        
        result = obfuscator._get_route_to_root(mock_token)
        assert result == [("root", "L")] or result == [("root", "R")] # Direction doesn't matter for root
    
    def test_get_route_to_root_child_token(self, obfuscator):
        """Test route to root for a child token"""
        mock_root = Mock()
        mock_root.text = "root"
        mock_root.i = 1
        mock_root.head = mock_root
        
        mock_child = Mock()
        mock_child.text = "child"
        mock_child.i = 0
        mock_child.head = mock_root
        
        result = obfuscator._get_route_to_root(mock_child)
        assert len(result) == 2
        assert result[0] == ("root", "R")  # Child is to the left of root
        assert result[1] == ("child", "L")
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
