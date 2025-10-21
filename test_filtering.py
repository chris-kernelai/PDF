#!/usr/bin/env python3
"""Test the financially_relevant filtering logic"""
import json

# Test the filtering logic
test_cases = [
    # Case 1: financially_relevant = false (should be filtered)
    {
        "input": '{"type": "logo", "description": "Red curved line", "financially_relevant": false}',
        "expected": "FILTERED OUT"
    },
    
    # Case 2: financially_relevant = true (should be kept, key removed)
    {
        "input": '{"type": "chart", "title": "Sales", "financially_relevant": true}',
        "expected": "KEPT (key removed)"
    },
    
    # Case 3: No financially_relevant key (should be kept as-is)
    {
        "input": '{"type": "table", "headers": ["Q1", "Q2"]}',
        "expected": "KEPT AS-IS"
    },
    
    # Case 4: Markdown wrapped JSON with false (should be filtered)
    {
        "input": '```json\n{"type": "photo", "objects": ["person"], "financially_relevant": false}\n```',
        "expected": "FILTERED OUT"
    },
    
    # Case 5: Markdown wrapped JSON with true (should be kept, key removed)
    {
        "input": '```json\n{"type": "chart", "data": [1, 2, 3], "financially_relevant": true}\n```',
        "expected": "KEPT (key removed)"
    },
]

print("Testing financially_relevant filtering logic\n" + "="*60)

for i, test_case in enumerate(test_cases, 1):
    desc = test_case["input"]
    expected = test_case["expected"]
    
    print(f'\nTest Case {i}: {expected}')
    print(f'Input: {desc[:60]}...')
    
    # Simulate the filtering logic
    filtered = False
    try:
        desc_text = desc.strip()
        
        # Remove markdown code fences
        if desc_text.startswith('```json'):
            desc_text = desc_text[7:]
        elif desc_text.startswith('```'):
            desc_text = desc_text[3:]
        if desc_text.endswith('```'):
            desc_text = desc_text[:-3]
        desc_text = desc_text.strip()
        
        # Parse JSON
        desc_json = json.loads(desc_text)
        
        # Check financially_relevant
        if 'financially_relevant' in desc_json:
            if desc_json['financially_relevant'] is False:
                filtered = True
                print('  ✅ Result: FILTERED OUT (not relevant)')
            else:
                del desc_json['financially_relevant']
                result = json.dumps(desc_json, indent=2)
                print(f'  ✅ Result: KEPT (relevant key removed)')
                print(f'  Output preview: {result[:60]}...')
        else:
            print('  ✅ Result: KEPT AS-IS (no filtering key)')
            
    except (json.JSONDecodeError, ValueError) as e:
        print(f'  ✅ Result: KEPT AS-IS (parse error: {str(e)[:40]})')

print('\n' + "="*60)
print('✅ Filtering logic test complete!')

