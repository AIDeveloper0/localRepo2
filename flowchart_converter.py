"""
Enhanced Flowchart Converter with Advanced NLP
Converts natural language descriptions into Mermaid flowchart diagrams
Uses spaCy for improved language understanding and pattern extraction
"""
import spacy
import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFlowchartConverter:
    """
    Advanced NLP-based flowchart converter using spaCy for superior pattern recognition
    """
    
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded successfully")
        except OSError:
            logger.error("❌ spaCy model not found - installing...")
            import subprocess
            try:
                subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy model installed and loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to install spaCy model: {str(e)}")
                self.nlp = None
        
        # Dynamic patterns for different types of processes
        self.conditional_keywords = ['if', 'when', 'whenever', 'should', 'must', 'check', 'verify', 'validate']
        self.action_keywords = ['then', 'process', 'send', 'update', 'generate', 'create', 'notify', 'charge']
        self.alternative_keywords = ['else', 'otherwise', 'if not', 'unless', 'except', 'alternative']
        self.sequence_keywords = ['then', 'next', 'after', 'subsequently', 'following', 'and then']
        
        self.node_counter = 0
        self.edge_counter = 0
        self.error_occurred = False
        self.last_error = None
        
        logger.info("✅ EnhancedFlowchartConverter initialized")

    def convert(self, text_input: str) -> str:
        """
        Main public method to convert text to flowchart
        
        Args:
            text_input: The English text to convert
            
        Returns:
            str: Mermaid DSL code
        """
        try:
            self.error_occurred = False
            self.last_error = None
            
            logger.info(f"Processing flowchart text: {text_input}")
            
            # Clean and normalize input text
            text_input = text_input.strip()
            if not text_input:
                logger.warning("Empty input text provided")
                return self._create_simple_flowchart()
            
            # Process text to flowchart data
            flowchart_data = self.convert_text_to_flowchart(text_input)
            
            # Validate Mermaid syntax before returning
            mermaid_code = flowchart_data["mermaid"]
            if "None -->" in mermaid_code:
                logger.warning("Invalid 'None' node detected in Mermaid output, fixing...")
                mermaid_code = mermaid_code.replace("None -->", "start -->")
            
            # Additional validation to ensure proper line breaks between node definitions
            lines = mermaid_code.split('\n')
            fixed_lines = []
            
            # Ensure graph declaration is present
            if not any(line.strip() == "graph TD" for line in lines):
                fixed_lines.append("graph TD")
            
            for i, line in enumerate(lines):
                if line.strip() == "graph TD" and i > 0:
                    # Don't add duplicate graph declarations
                    continue
                    
                # Check for multiple node definitions in a single line
                if line.strip() and ']' in line and '[' in line.split(']', 1)[1]:
                    parts = re.findall(r'(\S+\[.*?\]|\S+\{.*?\})', line)
                    for part in parts:
                        fixed_lines.append(part)
                # Check for node and edge definition on the same line
                elif line.strip() and (']' in line or '}' in line) and '-->' in line:
                    node_part = line.split('-->', 1)[0].strip()
                    edge_part = '-->' + line.split('-->', 1)[1].strip()
                    fixed_lines.append(node_part)
                    fixed_lines.append(edge_part)
                else:
                    fixed_lines.append(line)
            
            # Reconstruct the mermaid code with proper line breaks
            mermaid_code = '\n'.join(fixed_lines)
            
            # Final format verification
            # Ensure all nodes and edges are on separate lines with proper indentation
            lines = mermaid_code.split('\n')
            final_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    final_lines.append("")
                elif line == "graph TD":
                    final_lines.append("graph TD")
                elif line.startswith("classDef"):
                    final_lines.append("    " + line)
                elif ("[" in line or "{" in line) and not "-->" in line:
                    # Node definition
                    final_lines.append("    " + line)
                elif "-->" in line:
                    # Edge definition
                    final_lines.append("    " + line)
                else:
                    # Other content
                    final_lines.append(line)
            
            mermaid_code = '\n'.join(final_lines)
            
            return mermaid_code

        except Exception as e:
            self.error_occurred = True
            self.last_error = str(e)
            logger.error(f"❌ Error in flowchart conversion: {str(e)}")
            
            # Return fallback flowchart
            return """graph TD
    A["Start"] --> B["Process"]
    B --> C["End"]
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px"""

    def _create_simple_flowchart(self) -> str:
        """Create a simple default flowchart for edge cases"""
        return """graph TD
    start["Start"] --> process["Process"]
    process --> end["End"]
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px"""

    def _extract_sentence_structure(self, text: str) -> Dict[str, Any]:
        """Dynamically extract sentence structure using spaCy NLP with improved sequence handling"""
        if not self.nlp:
            return self._fallback_structure_extraction(text)
        
        doc = self.nlp(text)
        
        structure = {
            'subjects': [],
            'verbs': [],
            'objects': [],
            'conditions': [],
            'actions': [],
            'entities': [],
            'sentences': []
        }
        
        # Extract sentences
        for sent in doc.sents:
            structure['sentences'].append(sent.text.strip())
        
        # Extract linguistic elements
        for token in doc:
            # Subjects (who/what is acting)
            if token.dep_ in ['nsubj', 'nsubjpass']:
                structure['subjects'].append(token.text.lower())
            
            # Verbs (actions)
            if token.pos_ == 'VERB' and not token.is_stop:
                structure['verbs'].append(token.lemma_.lower())
            
            # Objects (what's being acted upon)
            if token.dep_ in ['dobj', 'pobj', 'obj']:
                structure['objects'].append(token.text.lower())
        
        # Extract named entities
        for ent in doc.ents:
            structure['entities'].append({
                'text': ent.text,
                'label': ent.label_,
                'type': spacy.explain(ent.label_)
            })
        
        # Integrate sequential actions directly here
        sequential_actions = self._extract_sequential_actions_from_text(text)
        if sequential_actions:
            # Group actions by sentence to maintain proper sequence
            structure['sequences'] = [{'actions': [a['action'] for a in sequential_actions]}]
        else:
            # Fallback to empty sequences
            structure['sequences'] = []
        
        logger.info(f"🧠 Extracted structure: {len(structure['sentences'])} sentences, {len(structure['verbs'])} verbs")
        logger.info(f"🧠 Found {len(structure.get('sequences', [])[0]['actions']) if structure.get('sequences') else 0} sequential actions")
        
        return structure

    def _identify_process_patterns(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically identify different process patterns in the text"""
        
        patterns = {
            'conditionals': [],
            'sequences': [],
            'parallels': [],
            'loops': [],
            'decisions': []
        }
        
        text_lower = text.lower()
        sentences = structure['sentences']
        
        # Identify conditional patterns
        for i, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            
            # Find IF-THEN patterns
            if any(keyword in sent_lower for keyword in self.conditional_keywords):
                condition_match = self._extract_if_then_pattern(sentence)
                if condition_match:
                    patterns['conditionals'].append({
                        'sentence_index': i,
                        'condition': condition_match['condition'],
                        'then_action': condition_match['then_action'],
                        'else_action': condition_match.get('else_action'),
                        'original': sentence
                    })
            
            # Find sequential patterns
            if any(keyword in sent_lower for keyword in self.sequence_keywords):
                patterns['sequences'].append({
                    'sentence_index': i,
                    'sequence_type': self._identify_sequence_type(sentence),
                    'actions': self._extract_sequential_actions(sentence),
                    'original': sentence
                })
            
            # Find decision points
            if '?' in sentence or any(word in sent_lower for word in ['decide', 'choose', 'select', 'determine']):
                patterns['decisions'].append({
                    'sentence_index': i,
                    'decision': self._extract_decision_text(sentence),
                    'options': self._extract_decision_options(sentence),
                    'original': sentence
                })
        
        # Enhanced conditional detection for simple sentences
        sentences = structure['sentences']
        for i, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            
            # Check for very simple if/else structure not captured by keywords
            if sent_lower.startswith('if ') and ('otherwise' in sent_lower or 'else' in sent_lower):
                parts = re.split(r'otherwise|else', sent_lower, maxsplit=1)
                if len(parts) == 2:
                    if_part = parts[0].replace('if', '', 1).strip()
                    else_part = parts[1].strip()
                    
                    # Split the if part into condition and then-action
                    if ',' in if_part:
                        condition, then_action = if_part.split(',', 1)
                        patterns['conditionals'].append({
                            'sentence_index': i,
                            'condition': condition.strip(),
                            'then_action': then_action.strip(),
                            'else_action': else_part,
                            'original': sentence
                        })
        
        logger.info(f"🔍 Identified patterns: {len(patterns['conditionals'])} conditionals, {len(patterns['sequences'])} sequences, {len(patterns['decisions'])} decisions")
        return patterns

    def _extract_if_then_pattern(self, sentence: str) -> Optional[Dict[str, str]]:
        """Extract IF-THEN-ELSE pattern from a sentence"""
        
        patterns = [
            # Your existing patterns
            r'if\s+(.*?)\s+then\s+(.*?)(?:\s+(?:else|otherwise)\s+(.*?))?(?:\.|$)',
            r'when\s+(.*?),?\s*(.*?)(?:\s+(?:else|otherwise)\s+(.*?))?(?:\.|$)',
            r'should\s+(.*?),?\s*(.*?)(?:\s+(?:if not|unless)\s+(.*?))?(?:\.|$)',
            
            # Add this simple pattern that will match your specific example
            r'if\s+(.*?),\s+(.*?)\.?\s*(?:otherwise|else)[,\s]+(.*?)(?:\.|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sentence.lower(), re.IGNORECASE)
            if match:
                result = {
                    'condition': match.group(1).strip(),
                    'then_action': match.group(2).strip()
                }
                
                # Handle else action if present
                if len(match.groups()) >= 3 and match.group(3):
                    result['else_action'] = match.group(3).strip()
                
                # Log successful pattern match
                logger.info(f"✅ Extracted condition: '{result['condition']}', then: '{result['then_action']}', else: '{result.get('else_action', 'None')}'")
                return result
        
        logger.warning(f"❌ No conditional pattern found in: '{sentence}'")
        return None

    def _extract_sequential_actions(self, sentence: str) -> List[str]:
        """Extract sequential actions from a sentence"""
        
        # Split on common sequence separators
        separators = ['and then', 'then', 'next', 'after that', 'subsequently', 'and']
        
        actions = [sentence]
        for sep in separators:
            new_actions = []
            for action in actions:
                new_actions.extend([part.strip() for part in action.split(sep) if part.strip()])
            actions = new_actions
        
        # Clean up actions
        cleaned_actions = []
        for action in actions:
            action = re.sub(r'^(and\s+|then\s+|next\s+)', '', action.strip())
            if action and len(action) > 3:
                cleaned_actions.append(action)
        
        return cleaned_actions[:6]  # Limit to prevent overly complex diagrams

    def _identify_sequence_type(self, sentence: str) -> str:
        """Identify the type of sequence (linear, parallel, conditional)"""
        
        sent_lower = sentence.lower()
        
        if any(word in sent_lower for word in ['simultaneously', 'parallel', 'at the same time', 'concurrently']):
            return 'parallel'
        elif any(word in sent_lower for word in ['if', 'when', 'should', 'depending']):
            return 'conditional'
        else:
            return 'linear'

    def _extract_decision_text(self, sentence: str) -> str:
        """Extract decision text from a sentence"""
        
        # Remove question marks and clean up
        decision = sentence.replace('?', '').strip()
        
        # Extract the core decision
        if 'whether' in decision.lower():
            match = re.search(r'whether\s+(.*)', decision.lower())
            if match:
                decision = match.group(1).strip()
        
        return decision[:80]  # Limit length for diagram readability

    def _extract_decision_options(self, sentence: str) -> List[str]:
        """Extract decision options from a sentence"""
        
        # Look for explicit options
        options = []
        
        # Check for "or" patterns
        if ' or ' in sentence.lower():
            parts = sentence.split(' or ')
            options = [part.strip() for part in parts if part.strip()]
        
        # Default yes/no for questions
        if '?' in sentence and not options:
            options = ['Yes', 'No']
        
        return options[:4]  # Limit options for diagram clarity

    def _extract_process_steps(self, text: str) -> List[Dict[str, str]]:
        """Extract sequential process steps from text"""
        
        steps = []
        
        # Split into sentences
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for i, sentence in enumerate(sentences):
            # Check for "When X, Y" pattern
            when_match = re.search(r'when\s+(.*?),\s+(.*)', sentence, re.IGNORECASE)
            if when_match:
                steps.append({
                    'type': 'event',
                    'trigger': when_match.group(1).strip(),
                    'action': when_match.group(2).strip()
                })
                continue
            
            # Check for "If X, Y; else Z" pattern
            if_match = re.search(r'if\s+(.*?),\s+(.*?)(?:;|$)\s*(?:else|otherwise),?\s*(.*?)(?:;|$)', 
                                sentence, re.IGNORECASE)
            if if_match:
                steps.append({
                    'type': 'condition',
                    'condition': if_match.group(1).strip(),
                    'then_action': if_match.group(2).strip(),
                    'else_action': if_match.group(3).strip() if if_match.group(3) else None
                })
                continue
        
        return steps

    def _extract_sequential_actions_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract sequences of actions from text using NLP and regex patterns"""
        if not self.nlp:
            return []
        
        sequential_actions = []
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # First extract sentence-level actions
        for sent_idx, sent in enumerate(doc.sents):
            # Skip sentences that are clearly conditional
            if any(keyword in sent.text.lower() for keyword in ['if ', 'when ', 'should ']):
                continue
                
            sent_text = sent.text.strip()
            
            # Add explicit pattern for "verb[s] noun" before commas
            comma_actions = re.findall(r'(\w+s(?:es)?\s+[^,;.]+)(?:,|;|and\s+)', sent_text, re.IGNORECASE)
            for action in comma_actions:
                if action.strip() and len(action.strip()) > 3:
                    sequential_actions.append({
                        'action': action.strip(),
                        'sentence_idx': sent_idx,
                        'source': 'comma_pattern'
                    })
            
            # Look for specific verb+object pairs using spaCy
            for token in sent:
                if token.pos_ == 'VERB' and not token.is_stop:
                    # Find direct objects of this verb
                    obj_text = ""
                    for child in token.children:
                        if child.dep_ in ['dobj', 'pobj']:
                            obj_text = child.text
                            # Include any adjectives modifying the object
                            for grandchild in child.children:
                                if grandchild.dep_ == 'amod':
                                    obj_text = f"{grandchild.text} {obj_text}"
                            break
                    
                    if obj_text:
                        action = f"{token.text} {obj_text}"
                        # Avoid duplicates
                        if not any(a['action'].lower() == action.lower() for a in sequential_actions):
                            sequential_actions.append({
                                'action': action,
                                'sentence_idx': sent_idx,
                                'source': 'spacy_verb_obj'
                            })
        
        return sequential_actions

    def _create_dynamic_flowchart(self, text: str, structure: Dict = None, patterns: Dict = None, sequential_actions: List = None) -> Dict[str, Any]:
        """Create flowchart with guaranteed sequence handling"""
        
        logger.info(f"🧠 Creating dynamic flowchart from text analysis...")
        
        # Use provided structure/patterns or create them if not provided
        if structure is None:
            structure = self._extract_sentence_structure(text)
        
        if patterns is None:
            patterns = self._identify_process_patterns(text, structure)
        
        # Build flowchart dynamically
        nodes = []
        edges = []
        
        # Create start node
        start_node = self._create_node('start', "Start", node_type='input')
        nodes.append(start_node)
        
        current_node_id = 'start'
        y_position = 150
        
        # CRITICAL FIX: Always check structure sequences first, then fall back to sequential_actions
        sequence_actions = []
        if structure.get('sequences') and structure['sequences'][0].get('actions'):
            sequence_actions = structure['sequences'][0]['actions']
        elif sequential_actions:
            sequence_actions = [a['action'] for a in sequential_actions]
        
        # Log that we're actually processing sequences
        logger.info(f"🔄 Processing {len(sequence_actions)} sequential actions in flowchart creation")
        
        # Process sequential actions first from structure.sequences
        if sequence_actions:
            logger.info(f"Processing {len(sequence_actions)} sequential actions")
            
            for i, action_text in enumerate(sequence_actions):
                action_id = f"action_{i+1}"
                
                # Capitalize first letter for better presentation
                if action_text and action_text[0].islower():
                    action_text = action_text[0].upper() + action_text[1:]
                
                action_node = self._create_node(
                    action_id,
                    action_text,
                    position={'x': 400, 'y': y_position}
                )
                nodes.append(action_node)
                
                # Connect to previous node
                edges.append(self._create_edge(current_node_id, action_id))
                
                # Update for next iteration
                current_node_id = action_id
                y_position += 100
    
        # Step 4: Process conditionals (existing logic)
        for i, conditional in enumerate(patterns.get('conditionals', [])):
            decision_id = f"decision_{i+1}"
            success_id = f"success_{i+1}"
            failure_id = f"failure_{i+1}"
            
            # Create decision node with better formatting
            condition_text = conditional['condition'].strip()
            if not condition_text.endswith('?'):
                condition_text += '?'
                
            decision_node = self._create_node(
                decision_id,
                f"Is {condition_text}",
                node_type='decision',
                position={'x': 400, 'y': y_position}
            )
            nodes.append(decision_node)
            
            # Create success path with better text formatting
            then_action = conditional['then_action'].strip()
            if then_action and then_action[0].islower():
                then_action = then_action[0].upper() + then_action[1:]
                
            success_node = self._create_node(
                success_id,
                then_action,
                position={'x': 200, 'y': y_position + 150}
            )
            nodes.append(success_node)
            
            # Connect previous node to the decision
            edges.append(self._create_edge(current_node_id, decision_id))
            
            # Connect decision to success path
            edges.append(self._create_edge(decision_id, success_id, label="YES"))
            
            # Create failure path if exists
            if conditional.get('else_action'):
                else_action = conditional['else_action'].strip()
                if else_action and else_action[0].islower():
                    else_action = else_action[0].upper() + else_action[1:]
                    
                failure_node = self._create_node(
                    failure_id,
                    else_action,
                    position={'x': 600, 'y': y_position + 150}
                )
                nodes.append(failure_node)
                
                # Add failure edge
                edges.append(self._create_edge(decision_id, failure_id, label="NO"))
                
                # Both paths continue to next step or end
                if i == len(patterns.get('conditionals', [])) - 1:
                    # If last conditional, connect both branches to end
                    edges.append(self._create_edge(success_id, 'end'))
                    edges.append(self._create_edge(failure_id, 'end')) 
                    # Don't update current_node_id as we're already connected to end
                else:
                    # Find a merge point - both branches connect to a new node
                    merge_id = f"merge_{i+1}"
                    merge_node = self._create_node(
                        merge_id,
                        "Continue",
                        position={'x': 400, 'y': y_position + 300}
                    )
                    nodes.append(merge_node)
                    edges.append(self._create_edge(success_id, merge_id))
                    edges.append(self._create_edge(failure_id, merge_id))
                    current_node_id = merge_id
            else:
                # No else path, just continue from the success path
                current_node_id = success_id
                
            y_position += 300
        
        # Add end node dynamically
        end_node = self._create_node(
            'end',
            "End",
            node_type='output',
            position={'x': 400, 'y': y_position}
        )
        nodes.append(end_node)
        
        # Connect the last node to the end if not already connected
        if not any(edge['target'] == 'end' for edge in edges):
            edges.append(self._create_edge(current_node_id, 'end'))
        
        logger.info(f"✅ Created dynamic flowchart with {len(nodes)} nodes and {len(edges)} edges")
        return {'nodes': nodes, 'edges': edges}

    def _create_node(self, node_id: str, label: str, node_type: str = 'default', position: Dict[str, int] = None) -> Dict[str, Any]:
        """Create a node with dynamic properties"""
        
        if position is None:
            position = {'x': 400, 'y': self.node_counter * 100 + 50}
        
        node = {
            'id': node_id,
            'data': {'label': label},
            'position': position
        }
        
        if node_type in ['input', 'output']:
            node['type'] = node_type
        
        self.node_counter += 1
        return node

    def _create_edge(self, source: str, target: str, label: str = None) -> Dict[str, Any]:
        """Create an edge with optional label"""
        
        edge = {
            'id': f"e_{self.edge_counter}",
            'source': source,
            'target': target
        }
        
        if label:
            edge['label'] = label
        
        self.edge_counter += 1
        return edge

    def _fallback_structure_extraction(self, text: str) -> Dict[str, Any]:
        """Fallback structure extraction when spaCy is not available"""
        
        sentences = text.split('.')
        words = text.lower().split()
        
        return {
            'subjects': [w for w in words if w in ['user', 'customer', 'system', 'process', 'order']],
            'verbs': [w for w in words if w in ['process', 'validate', 'send', 'update', 'create', 'check']],
            'objects': [w for w in words if w in ['order', 'payment', 'email', 'data', 'information']],
            'sentences': [s.strip() for s in sentences if s.strip()],
            'entities': []
        }

    def _sanitize_node_id(self, node_id: str) -> str:
        """Sanitize node IDs to avoid Mermaid reserved keywords"""
        # List of Mermaid reserved keywords
        reserved_keywords = ['end', 'graph', 'subgraph', 'class', 'click', 'style', 'linkStyle']
        
        # If node_id is a reserved keyword, append "_node" to it
        if node_id.lower() in reserved_keywords:
            logger.warning(f"⚠️ Found reserved Mermaid keyword '{node_id}' as node ID, renaming to '{node_id}_node'")
            return f"{node_id}_node"
        return node_id

    def convert_to_mermaid(self, flowchart_data: Dict[str, Any]) -> str:
        """Convert to Mermaid with guaranteed correct syntax and better label handling"""
        try:
            # Start with proper graph declaration
            mermaid_parts = ["graph TD"]
            
            # Process all nodes with explicit newlines
            for node in flowchart_data.get('nodes', []):
                # Sanitize the node ID to avoid reserved keywords
                original_id = node['id']
                node_id = self._sanitize_node_id(original_id)
                
                label = node['data']['label']
                
                # Clean label thoroughly
                clean_label = str(label).replace('\n', ' ').replace('"', "'").strip()
                clean_label = re.sub(r'[^\w\s.,;:!?()\'"-]', '', clean_label)
                
                if not clean_label:
                    clean_label = f"Node_{node_id}"
                # Allow up to 40 characters for better readability (increased from original)
                elif len(clean_label) > 40:
                    clean_label = clean_label[:37] + "..."
                
                # Generate node with proper syntax - each node on its own line
                if node.get('type') == 'input' or 'start' in original_id.lower():
                    mermaid_parts.append(f"    {node_id}[\"{clean_label}\"]")
                elif node.get('type') == 'output' or 'end' in original_id.lower():
                    mermaid_parts.append(f"    {node_id}[\"{clean_label}\"]")
                elif 'decision' in original_id.lower() or node.get('type') == 'decision':
                    mermaid_parts.append(f"    {node_id}{{\"{clean_label}\"}}")
                else:
                    mermaid_parts.append(f"    {node_id}[\"{clean_label}\"]")
            
            # Store node ID mapping for edges
            id_mapping = {node['id']: self._sanitize_node_id(node['id']) 
                        for node in flowchart_data.get('nodes', [])}
            
            # Add blank line for clarity
            mermaid_parts.append("")
            
            # Process all edges with proper ID references
            for edge in flowchart_data.get('edges', []):
                source = id_mapping.get(edge['source'], edge['source'])
                target = id_mapping.get(edge['target'], edge['target'])
                
                # Skip edges with None as source or target
                if source is None or target is None or source == "None" or target == "None":
                    continue
                    
                label = edge.get('label', '')
                
                if label and label.strip():
                    clean_edge_label = str(label).replace('"', "'").strip()
                    mermaid_parts.append(f"    {source} -->|{clean_edge_label}| {target}")
                else:
                    mermaid_parts.append(f"    {source} --> {target}")
            
            # Add styling with proper spacing
            mermaid_parts.append("")
            mermaid_parts.append("    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px")
            mermaid_parts.append("    classDef startEnd fill:#e8f5e8,stroke:#4caf50,stroke-width:2px")
            
            # Join with explicit newlines
            mermaid_code = "\n".join(mermaid_parts)
            
            logger.info(f"✅ Generated Mermaid code with {len(mermaid_parts)} lines")
            return mermaid_code
        
        except Exception as e:
            logger.error(f"❌ Error in Mermaid conversion: {str(e)}")
            return """graph TD
    start["Start"] --> process["Process"]
    process --> finish["End"]
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px"""

    def convert_text_to_flowchart(self, text_input: str) -> Dict[str, Any]:
        """Single-pass, unit-based conversion: handles sequences and conditionals in order"""
        import re
        logger.info(f"🚀 SINGLE-PASS CONVERSION: {len(text_input)} chars")
        self.node_counter = 0
        self.edge_counter = 0

        def make_node_id(prefix):
            self.node_counter += 1
            return f"{prefix}{self.node_counter}"

        def make_node(node_id, label, node_type=None, position=None):
            node = {'id': node_id, 'data': {'label': label}}
            if node_type:
                node['type'] = node_type
            if position:
                node['position'] = position
            return node

        def render_mermaid(nodes, edges):
            parts = ["graph TD"]
            
            # Process nodes - ensure each node is on its own line
            for node in nodes:
                nid = node['id']
                label = node['data']['label']
                # Sanitize label: replace quotes, remove newlines, strip whitespace
                clean_label = str(label).replace('"', "'").replace('\n', ' ').strip()
                clean_label = clean_label.strip("'").strip('"')
                # Remove trailing periods followed by quotes (edge case)
                if clean_label.endswith(".'") or clean_label.endswith('."'):
                    clean_label = clean_label[:-2] + "'"
                
                if node.get('type') == 'decision' or 'decision' in nid:
                    parts.append(f'    {nid}{{"{clean_label}"}}')
                else:
                    parts.append(f'    {nid}["{clean_label}"]')
            
            # Add blank line for clarity
            parts.append("")
            
            # Process all edges with proper ID references and avoid "None" node IDs
            for src, tgt, lbl in edges:
                # Skip edges with None as source or target
                if src is None or tgt is None or src == "None" or tgt == "None":
                    continue
                    
                if lbl:
                    parts.append(f"    {src} -->|{lbl}| {tgt}")
                else:
                    parts.append(f"    {src} --> {tgt}")
            
            # Add styling
            parts.append("")
            parts.append("    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px")
            parts.append("    classDef startEnd fill:#e8f5e8,stroke:#4caf50,stroke-width:2px")
            
            # Join with explicit newlines and ensure newlines between sections
            return "\n".join(parts)

        def extract_sequential_actions(unit):
            # Use spaCy if available, fallback to regex
            actions = []
            if self.nlp:
                doc = self.nlp(unit)
                for sent in doc.sents:
                    # Skip conditionals
                    if any(k in sent.text.lower() for k in ['if ', 'when ', 'else ', 'otherwise ']):
                        continue
                    # Try to extract verb-object pairs
                    for token in sent:
                        if token.pos_ == "VERB" and not token.is_stop:
                            obj = ""
                            for child in token.children:
                                if child.dep_ in ['dobj', 'pobj']:
                                    obj = child.text
                                    for grandchild in child.children:
                                        if grandchild.dep_ == 'amod':
                                            obj = f"{grandchild.text} {obj}"
                                    break
                            if obj:
                                action = f"{token.lemma_} {obj}".strip()
                                if not any(a.lower() == action.lower() for a in actions):
                                    actions.append(action)
            # Regex fallback: split on commas, and, then
            if not actions:
                for part in re.split(r',\s*|\s+and\s+|\s+then\s+', unit):
                    if part.strip() and len(part.strip()) > 5:
                        clean_part = part.strip()
                        if re.search(r'\b(add|browse|select|check|view|process|submit|enter|use|click)\w*\b', clean_part, re.IGNORECASE):
                            actions.append(clean_part)
            # Final fallback: just split on commas
            if not actions:
                actions = [a.strip() for a in re.split(r',| and ', unit) if a.strip()]
            return actions

        def extract_if_else(unit):
            # Try direct pattern: if X, Y. otherwise Z.
            m = re.search(r'if\s+(.*?)\s*,\s*(.*?)\.\s*otherwise[, ]+(.*?)\.?$', unit, re.IGNORECASE)
            if m:
                return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            m = re.search(r'if\s+(.*?)\s*,\s*(.*?)\.\s*else[, ]+(.*?)\.?$', unit, re.IGNORECASE)
            if m:
                return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            m = re.search(r'if\s+(.*?)\s*,\s*(.*?)\s*(?:otherwise|else)[, ]+(.*?)\.?$', unit, re.IGNORECASE)
            if m:
                return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            # Try pattern matching
            if hasattr(self, '_extract_if_then_pattern'):
                cond_match = self._extract_if_then_pattern(unit)
                if cond_match:
                    return cond_match['condition'], cond_match['then_action'], cond_match.get('else_action', '')
            return unit, '', ''

        def split_into_units(text):
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            units, current = [], []
            for s in sentences:
                if current and re.match(r'^(otherwise|else)\b', s, re.IGNORECASE):
                    current.append(s)
                else:
                    if current:
                        units.append(" ".join(current))
                    current = [s]
            if current: units.append(" ".join(current))
            return units

        # --- Main pipeline ---
        units = split_into_units(text_input)
        nodes, edges = [], []
        nodes.append(make_node("start", "Start", node_type='input'))
        prev = "start"
        branch_ends = []
        for unit in units:
            if unit.lower().startswith("if") or re.match(r'^if ', unit, re.IGNORECASE):
                cond, then_act, else_act = extract_if_else(unit)
                dec = make_node_id("decision")
                then_n = make_node_id("then")
                else_n = make_node_id("else")
                nodes += [
                    make_node(dec, f"Is {cond}?", node_type='decision'),
                    make_node(then_n, then_act or "Process"),
                    make_node(else_n, else_act or "Alternative"),
                ]
                edges += [
                    (prev, dec, None),
                    (dec, then_n, "Yes"),
                    (dec, else_n, "No"),
                ]
                prev = None
                branch_ends = [then_n, else_n]
            else:
                actions = extract_sequential_actions(unit)
                for act in actions:
                    node_id = make_node_id("step")
                    nodes.append(make_node(node_id, act))
                    edges.append((prev, node_id, None))
                    prev = node_id
        nodes.append(make_node("end", "End", node_type='output'))
        if prev:
            edges.append((prev, "end", None))
        elif branch_ends:
            # Use list comprehension to filter out None values
            valid_branch_ends = [b for b in branch_ends if b is not None]
            for b in valid_branch_ends:
                edges.append((b, "end", None))
        mermaid = render_mermaid(nodes, edges)
        
        return {
            "mermaid": mermaid,
            "nodes": nodes,
            "edges": edges
        }
        
    def _identify_pattern_type(self, text: str) -> str:
        """Identify the type of flowchart pattern in the text"""
        text_lower = text.lower()
        
        # Decision pattern (if-then-else)
        if (re.search(r'if\s+.+?\s+then\s+.+?\s+(?:else|otherwise)', text_lower, re.IGNORECASE) or
            re.search(r'if\s+.+?\s+,\s+.+?\s+(?:else|otherwise)', text_lower, re.IGNORECASE)):
            return "decision"
            
        # Nested decision pattern
        if (re.search(r'if\s+.+?\s+then\s+.+?\s+if\s+', text_lower, re.IGNORECASE) or
            re.search(r'if\s+.+?\s+,\s+.+?\s+if\s+', text_lower, re.IGNORECASE)):
            return "nested_decision"
            
        # While loop pattern
        if (re.search(r'while\s+.+?,.+?', text_lower, re.IGNORECASE) or
            re.search(r'as long as\s+.+?,.+?', text_lower, re.IGNORECASE)):
            return "while_loop"
            
        # Loop pattern
        if (re.search(r'repeat\s+.+?\s+until\s+', text_lower, re.IGNORECASE) or
            re.search(r'loop\s+through\s+', text_lower, re.IGNORECASE)):
            return "loop"
            
        # Parallel pattern
        if (re.search(r'simultaneously|in parallel|at the same time', text_lower, re.IGNORECASE) or
            re.search(r'both|concurrently', text_lower, re.IGNORECASE)):
            return "parallel"
            
        # Sequence pattern
        if (re.search(r'first.+?then|initially.+?next|begin by.+?after', text_lower, re.IGNORECASE) or
            len(re.findall(r',\s*then|,\s*next|,\s*finally', text_lower)) > 0 or
            len(text_lower.split(',')) > 2):
            return "sequence"
            
        # Default to sequence for simple lists
        return "sequence"
    
    def _process_decision_pattern(self, text: str) -> Dict[str, Any]:
        """Process a simple decision pattern"""
        # Extract conditional components
        match = re.search(r'(.+?)\s+if\s+(.+?)\s+then\s+(.+?)\s+(?:else|otherwise)\s+(.+?)$', 
                         text, re.IGNORECASE)
        
        if not match:
            # Try alternative pattern
            match = re.search(r'if\s+(.+?)\s*,\s*(.+?)\s*(?:else|otherwise)[,\s]+(.+?)$', 
                             text, re.IGNORECASE)
            if match:
                process = "Process"
                condition = self._clean_condition(match.group(1))
                success_action = self._clean_label(match.group(2))
                failure_action = self._clean_label(match.group(3))
            else:
                return self._create_fallback_flowchart(text)
        else:
            process = self._clean_label(match.group(1))
            condition = self._clean_condition(match.group(2))
            success_action = self._clean_label(match.group(3))
            failure_action = self._clean_label(match.group(4))
        
        # Generate mermaid code
        mermaid = f"""graph TD
    A["{process}"]
    B{{{condition}}}
    C["{success_action}"]
    D["{failure_action}"]
    A --> B
    B -->|Yes| C
    B -->|No| D"""
        
        # Create nodes and edges
        nodes = [
            {'id': 'A', 'data': {'label': process}},
            {'id': 'B', 'data': {'label': condition}, 'type': 'decision'},
            {'id': 'C', 'data': {'label': success_action}},
            {'id': 'D', 'data': {'label': failure_action}}
        ]
        
        edges = [
            {'id': 'e1', 'source': 'A', 'target': 'B'},
            {'id': 'e2', 'source': 'B', 'target': 'C', 'label': 'Yes'},
            {'id': 'e3', 'source': 'B', 'target': 'D', 'label': 'No'}
        ]
        
        return {
            "mermaid": mermaid,
            "nodes": nodes,
            "edges": edges
        }
        
    def _process_sequence_pattern(self, text: str) -> Dict[str, Any]:
        """Process a sequence pattern"""
        # Extract sequential steps
        steps = []
        
        # Try to find explicit sequence markers
        explicit_pattern = r'(?:first|initially|begin by|starts with|step 1)\s+(.+?)(?:\s+(?:then|next|after that|finally|lastly|eventually|step \d+)\s+)(.+)'
        match = re.search(explicit_pattern, text, re.IGNORECASE)
        
        if match:
            # Extract from explicit sequence markers
            steps.append(self._clean_label(match.group(1)))
            
            # Process remaining sequence
            remaining = match.group(2)
            remaining_steps = re.split(r'\s+(?:then|next|after that|finally|lastly|eventually|step \d+)\s+', 
                                     remaining, flags=re.IGNORECASE)
            steps.extend([self._clean_label(step) for step in remaining_steps if step.strip()])
        else:
            # Split by commas
            parts = [p.strip() for p in re.split(r',|\sand\s', text) if p.strip()]
            steps = [self._clean_label(part) for part in parts]
        
        # Generate mermaid code
        mermaid_lines = ["graph TD"]
        
        # Add nodes
        for i, step in enumerate(steps):
            node_id = chr(ord('A') + i)
            mermaid_lines.append(f'    {node_id}["{step}"]')
        
        # Add connections
        for i in range(len(steps) - 1):
            current = chr(ord('A') + i)
            next_node = chr(ord('A') + i + 1)
            mermaid_lines.append(f"    {current} --> {next_node}")
        
        mermaid = "\n".join(mermaid_lines)
        
        # Create nodes and edges
        nodes = []
        edges = []
        
        for i, step in enumerate(steps):
            node_id = chr(ord('A') + i)
            nodes.append({'id': node_id, 'data': {'label': step}})
            
            if i > 0:
                prev_id = chr(ord('A') + i - 1)
                edges.append({
                    'id': f'e{i}', 
                    'source': prev_id, 
                    'target': node_id
                })
        
        return {
            "mermaid": mermaid,
            "nodes": nodes,
            "edges": edges
        }
        
    def _process_loop_pattern(self, text: str) -> Dict[str, Any]:
        """Process a loop pattern"""
        # Try to match repeat-until pattern
        match = re.search(r'repeat\s+(.+?)\s+until\s+(.+)', text, re.IGNORECASE)
        
        if not match:
            # Try checking for items/collection iteration
            match = re.search(r'(?:for each|for all|loop through)\s+(.+?)\s+in\s+(.+?)[,.]?\s*(.+)', 
                             text, re.IGNORECASE)
            
        action = "Checking Status"
        condition = "Complete"
        
        if match:
            if len(match.groups()) >= 2:
                action = self._clean_label(match.group(1))
                condition = self._clean_condition(match.group(2))
        
        # Generate mermaid code
        mermaid = f"""graph TD
    A["Start"]
    B["{action}"]
    C{{{condition}?}}
    D["Continue"]
    A --> B
    B --> C
    C -->|No| B
    C -->|Yes| D"""
        
        # Create nodes and edges
        nodes = [
            {'id': 'A', 'data': {'label': 'Start'}},
            {'id': 'B', 'data': {'label': action}},
            {'id': 'C', 'data': {'label': f"{condition}?"}, 'type': 'decision'},
            {'id': 'D', 'data': {'label': 'Continue'}}
        ]
        
        edges = [
            {'id': 'e1', 'source': 'A', 'target': 'B'},
            {'id': 'e2', 'source': 'B', 'target': 'C'},
            {'id': 'e3', 'source': 'C', 'target': 'B', 'label': 'No'},
            {'id': 'e4', 'source': 'C', 'target': 'D', 'label': 'Yes'}
        ]
        
        return {
            "mermaid": mermaid,
            "nodes": nodes,
            "edges": edges
        }
        
    def _process_parallel_pattern(self, text: str) -> Dict[str, Any]:
        """Process a parallel pattern"""
        # Try to extract system and parallel processes
        match = re.search(r'(.+?)\s+(?:simultaneously|in parallel|at the same time|concurrently)\s+(.+?)\s+and\s+(.+)', 
                         text, re.IGNORECASE)
        
        system = "System"
        process1 = "Payment"
        process2 = "Inventory"
        
        if match:
            system = self._clean_label(match.group(1))
            process1 = self._clean_label(match.group(2))
            process2 = self._clean_label(match.group(3))
        else:
            # Try simpler pattern
            parts = text.split(' and ')
            if len(parts) >= 2:
                system_parts = parts[0].split(' ')
                if len(system_parts) > 2:
                    system = self._clean_label(' '.join(system_parts[:2]))
                    process1 = self._clean_label(' '.join(system_parts[2:]))
                    process2 = self._clean_label(parts[1])
                
        # Generate mermaid code
        mermaid = f"""graph TD
    A["{system}"]
    B["{process1}"]
    C["{process2}"]
    D["Complete"]
    A --> B
    A --> C
    B --> D
    C --> D"""
        
        # Create nodes and edges
        nodes = [
            {'id': 'A', 'data': {'label': system}},
            {'id': 'B', 'data': {'label': process1}},
            {'id': 'C', 'data': {'label': process2}},
            {'id': 'D', 'data': {'label': 'Complete'}}
        ]
        
        edges = [
            {'id': 'e1', 'source': 'A', 'target': 'B'},
            {'id': 'e2', 'source': 'A', 'target': 'C'},
            {'id': 'e3', 'source': 'B', 'target': 'D'},
            {'id': 'e4', 'source': 'C', 'target': 'D'}
        ]
        
        return {
            "mermaid": mermaid,
            "nodes": nodes,
            "edges": edges
        }
        
    def _process_nested_decision_pattern(self, text: str) -> Dict[str, Any]:
        """Process nested decision pattern"""
        # First check for the login pattern which is common and specific
        login_pattern = r'(.+?)\s+checks\s+(.+?)\s+if\s+(.+?)\s+then\s+(.+?)\s+(?:check|if)\s+(.+?)\s+then\s+(.+?)\s+(?:else|otherwise)\s+(.+?)\s+(?:else|otherwise)\s+(.+)'
        login_match = re.search(login_pattern, text, re.IGNORECASE)
        
        if login_match:
            process = self._clean_label(login_match.group(1))
            username_condition = self._clean_condition(login_match.group(3))
            password_condition = self._clean_condition(login_match.group(5))
            success_action = self._clean_label(login_match.group(6))
            failure_action = self._clean_label(login_match.group(7))
            username_error = self._clean_label(login_match.group(8))
            
            # Generate specific login flowchart
            mermaid = f"""graph TD
A["{process}"]
B{{"{username_condition}?"}}
C{{"{password_condition}?"}}
D["{success_action}"]
E["{failure_action}"]
F["{username_error}"]
A --> B
B -->|Yes| C
C -->|Yes| D
C -->|No| E
B -->|No| F"""
            
            # Create nodes and edges
            nodes = [
                {'id': 'A', 'data': {'label': process}},
                {'id': 'B', 'data': {'label': f"{username_condition}?"}, 'type': 'decision'},
                {'id': 'C', 'data': {'label': f"{password_condition}?"}, 'type': 'decision'},
                {'id': 'D', 'data': {'label': success_action}},
                {'id': 'E', 'data': {'label': failure_action}},
                {'id': 'F', 'data': {'label': username_error}}
            ]
            
            edges = [
                {'id': 'e1', 'source': 'A', 'target': 'B'},
                {'id': 'e2', 'source': 'B', 'target': 'C', 'label': 'Yes'},
                {'id': 'e3', 'source': 'C', 'target': 'D', 'label': 'Yes'},
                {'id': 'e4', 'source': 'C', 'target': 'E', 'label': 'No'},
                {'id': 'e5', 'source': 'B', 'target': 'F', 'label': 'No'}
            ]
            
            return {
                "mermaid": mermaid,
                "nodes": nodes,
                "edges": edges
            }
        
        # General nested decision pattern
        match = re.search(r'(.+?)\s+if\s+(.+?)\s+then\s+(.+?)\s+if\s+(.+?)\s+then\s+(.+?)(?:\s+(?:else|otherwise)\s+(.+?))?$', 
                     text, re.IGNORECASE)
    
        if not match:
            # Try system/process pattern
            match = re.search(r'(.+?)\s+(?:where|process|system)\s+(.+?)\s+if\s+(.+?)\s+then\s+(.+?)\s+if\s+(.+?)\s+then\s+(.+?)$', 
                             text, re.IGNORECASE)
        
        if not match:
            return self._create_simple_flowchart()
        
        # Extract components
        if len(match.groups()) >= 5:
            # Process where pattern
            if 'where' in match.group(0).lower() or 'system' in match.group(0).lower() or 'process' in match.group(0).lower():
                process = self._clean_label(f"{match.group(1)} {match.group(2)}")
                condition1 = self._clean_condition(match.group(3))
                action1 = self._clean_label(match.group(4))
                condition2 = self._clean_condition(match.group(5))
                action2 = self._clean_label(match.group(6)) if len(match.groups()) >= 6 else "Process"
            else:
                process = self._clean_label(match.group(1))
                condition1 = self._clean_condition(match.group(2))
                action1 = self._clean_label(match.group(3))
                condition2 = self._clean_condition(match.group(4))
                action2 = self._clean_label(match.group(5))
        
            # Generate mermaid code
            mermaid = f"""graph TD
A["{process}"]
B{{"{condition1}?"}}
C{{"{condition2}?"}}
D["{action2}"]
E["Alternative"]
A --> B
B -->|Yes| C
C -->|Yes| D
B -->|No| E"""
        
            # Create nodes and edges
            nodes = [
                {'id': 'A', 'data': {'label': process}},
                {'id': 'B', 'data': {'label': f"{condition1}?"}, 'type': 'decision'},
                {'id': 'C', 'data': {'label': f"{condition2}?"}, 'type': 'decision'},
                {'id': 'D', 'data': {'label': action2}},
                {'id': 'E', 'data': {'label': 'Alternative'}}
            ]
        
            edges = [
                {'id': 'e1', 'source': 'A', 'target': 'B'},
                {'id': 'e2', 'source': 'B', 'target': 'C', 'label': 'Yes'},
                {'id': 'e3', 'source': 'C', 'target': 'D', 'label': 'Yes'},
                {'id': 'e4', 'source': 'B', 'target': 'E', 'label': 'No'}
            ]
        
            return {
                "mermaid": mermaid,
                "nodes": nodes,
                "edges": edges
            }
    
        return self._create_simple_flowchart()
        
    def _process_while_loop_pattern(self, text: str) -> Dict[str, Any]:
        """Process while loop pattern"""
        # Try to match while pattern
        match = re.search(r'(?:while|as long as)\s+(.+?)[,.]?\s*(.+)', text, re.IGNORECASE)
        
        condition = "Unprocessed Items"
        action = "Process Next Item"
        
        if match:
            condition = self._clean_condition(match.group(1))
            action = self._clean_label(match.group(2))
        
        # Generate mermaid code
        mermaid = f"""graph TD
    A["Start"]
    B{{"{condition}?"}}
    C["{action}"]
    D["Finish"]
    A --> B
    B -->|Yes| C
    C --> B
    B -->|No| D"""
        
        # Create nodes and edges
        nodes = [
            {'id': 'A', 'data': {'label': 'Start'}},
            {'id': 'B', 'data': {'label': f"{condition}?"}, 'type': 'decision'},
            {'id': 'C', 'data': {'label': action}},
            {'id': 'D', 'data': {'label': 'Finish'}}
        ]
        
        edges = [
            {'id': 'e1', 'source': 'A', 'target': 'B'},
            {'id': 'e2', 'source': 'B', 'target': 'C', 'label': 'Yes'},
            {'id': 'e3', 'source': 'C', 'target': 'B'},
            {'id': 'e4', 'source': 'B', 'target': 'D', 'label': 'No'}
        ]
        
        return {
            "mermaid": mermaid,
            "nodes": nodes,
            "edges": edges
        }
    
    def _extract_main_process(self, text: str) -> str:
        """Extract main process name from text"""
        # Look for process/system keywords
        process_match = re.search(r'(.+?)\s+(?:system|process|workflow|procedure)', text, re.IGNORECASE)
        if process_match:
            return self._clean_label(process_match.group(1))
        
        # Take first meaningful phrase
        words = text.split()
        if len(words) >= 3:
            return self._clean_label(' '.join(words[:3]))
        else:
            return self._clean_label(text)
    
    def _clean_condition(self, condition_text: str) -> str:
        """Clean and format condition text"""
        # Remove question marks and normalize
        condition = condition_text.strip()
        if condition.endswith('?'):
            condition = condition[:-1]
            
        # Look for key condition indicators
        condition_keywords = {
            'successful': 'Successful',
            'valid': 'Valid', 
            'complete': 'Complete',
            'approved': 'Approved',
            'verified': 'Verified',
            'available': 'Available',
            'sufficient': 'Sufficient',
            'correct': 'Correct',
            'authenticated': 'Authenticated',
            'authorized': 'Authorized',
            'eligible': 'Eligible',
            'qualified': 'Qualified',
            'matching': 'Matching',
            'paid': 'Paid',
            'exists': 'Exists',
            'finished': 'Finished',
            'passes': 'Passes'
        }
        
        # Check for multiple conditions connected by 'and'
        if ' and ' in condition.lower():
            # Extract key conditions
            parts = condition.lower().split(' and ')
            keywords = []
            
            for part in parts:
                for keyword, label in condition_keywords.items():
                    if keyword in part.lower():
                        keywords.append(label)
                        break
                        
            if keywords:
                if len(keywords) == 1:
                    return keywords[0]
                else:
                    # Join up to 2 conditions with &
                    return " & ".join(keywords[:2])
            else:
                # No matched keywords, use the first part
                first_part = parts[0].strip().capitalize()
                return first_part if len(first_part) < 30 else first_part[:27] + "..."
        
        # Check for single condition
        for keyword, label in condition_keywords.items():
            if keyword in condition.lower():
                return label
        
        # Fallback: use first few words
        words = condition.strip().split()
        if len(words) >= 2:
            return f"{words[0].title()} {words[1].title()}"
        elif len(words) == 1:
            return f"{words[0].title()}"
        else:
            return "Complete"
    
    def _clean_label(self, text: str) -> str:
        """Clean text for node labels"""
        if not text:
            return "Process"
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common unnecessary words for better readability
        text = re.sub(r'\b(the|a|an|this|that)\b\s*', '', text, flags=re.IGNORECASE)
        
        # Handle long text by taking meaningful phrases
        if len(text) > 35:
            # Try to break at natural points
            if ',' in text:
                text = text.split(',')[0].strip()
            elif ' and ' in text:
                text = text.split(' and ')[0].strip()
            elif len(text.split()) > 5:
                # Take first 5 words
                words = text.split()
                text = ' '.join(words[:5])
        
        # Capitalize properly
        return text.strip().title() if text.strip() else "Process"

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the conversion process
        
        Returns:
            Dict[str, Any]: Statistics about the conversion
        """
        return {
            "nodes_created": self.node_counter,
            "error_occurred": self.error_occurred,
            "error_message": self.last_error if self.error_occurred else None
        }

# Factory function to create instance
def create_flowchart_converter():
    """Create and return an enhanced flowchart converter instance"""
    return EnhancedFlowchartConverter()

# Create a backward-compatible alias for DynamicFlowchartConverter
class DynamicFlowchartConverter(EnhancedFlowchartConverter):
    """
    Backward compatibility class for existing code that expects DynamicFlowchartConverter
    This class inherits all functionality from EnhancedFlowchartConverter
    """
    def __init__(self):
        super().__init__()
        logger.info("✅ DynamicFlowchartConverter initialized (alias for EnhancedFlowchartConverter)")
