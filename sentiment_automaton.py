from automata.fa.dfa import DFA
import matplotlib.pyplot as plt
import networkx as nx
from grammar_automaton import verificar_gramatica
import json
from pathlib import Path
from nltk.stem.snowball import SnowballStemmer
import spacy

class SentimentAutomaton:
    def __init__(self, lexicon_path: str = None):
        # Carrega modelo spaCy
        self.nlp = spacy.load("pt_core_news_sm")
        # Instancia SnowballStemmer para português
        self.stemmer = SnowballStemmer("portuguese")

        # Carrega o JSON de léxico
        lex_path = Path(lexicon_path or __file__).parent / "sentiment_words.json"
        with open(lex_path, encoding="utf-8") as f:
            data = json.load(f)

        # palavras que não serão consideradas
        self.stop_words = {';', '?', ',', '.', '!', ':', '-', '(', ')', '[', ']', '{', '}', "'", '"', '“', '”', '’',
                           'mas', 'e', 'ou', 'porque', 'porém', 'contudo', 'todavia', 'entretanto', 'embora', 'ainda',
                           'já', 'também', 'sim', 'simplesmente', 'apenas', 'só', 'mesmo', 'tão'}
        
        stem = self.stemmer.stem

        self.subs = { stem(w) for w in data.get("subs", []) }
        self.det = { stem(w) for w in data.get("det", []) }
        self.pron = { stem(w) for w in data.get("pron", []) }
        self.verbs = { stem(w) for w in data.get("verbs", []) }
        self.pos_verbs = { stem(w) for w in data.get("pos_verbs", []) }
        self.adj_pos = { stem(w) for w in data.get("adj_pos", []) }
        self.adj_neg = { stem(w) for w in data.get("adj_neg", []) }
        self.adj_neu = { stem(w) for w in data.get("adj_neu", []) }
        self.adv = { stem(w) for w in data.get("adv", []) }


        # Estados e símbolos
        states = {'q0', 'q1'}
        input_symbols = {'Det', 'Pron', 'Subs', 'Verb', 'PosVerb', 'AdjPos', 'AdjNeg', 'AdjNeu', 'Adv'}
        transitions = {
            'q0': {sym: 'q0' for sym in input_symbols},  
            'q1': {sym: 'q1' for sym in input_symbols}   
        }

        transitions['q0']['AdjPos'] = 'q1'
        transitions['q0']['AdjNeg'] = 'q1'
        transitions['q0']['AdjNeu'] = 'q1'
        transitions['q0']['PosVerb'] = 'q1'

        transitions['q1']['PosVerb'] = 'q1'
        transitions['q1']['AdjPos'] = 'q1'
        transitions['q1']['AdjNeg'] = 'q1'
        transitions['q1']['AdjNeu'] = 'q1'

        self.dfa = DFA(
            states=states,
            input_symbols=input_symbols,
            transitions=transitions,
            initial_state='q0',
            final_states={'q1'}
        )

    def tokenize_and_stem(self, sentence: str):
        doc = self.nlp(sentence.lower())
        # filtra pontuação e stop words
        filtered = [
            (token.text, token.lemma_)
            for token in doc
            if not token.is_punct and token.text not in self.stop_words
        ]
        if not filtered:
            return [], []

        tokens, lemmas = zip(*filtered)
        stems = [self.stemmer.stem(lemma) for lemma in lemmas]
        return list(tokens), list(stems)

    def classify(self, stems):
        classes = []
        for stem in stems:
            if stem in self.det:
                classes.append('Det')
            elif stem in self.pron:
                classes.append('Pron')
            elif stem in self.subs:
                classes.append('Subs')
            elif stem in self.verbs:
                classes.append('Verb')
            elif stem in self.pos_verbs:
                classes.append('PosVerb')
            elif stem in self.adj_pos:
                classes.append('AdjPos')
            elif stem in self.adj_neg:
                classes.append('AdjNeg')
            elif stem in self.adj_neu:
                classes.append('AdjNeu')
            elif stem in self.adv:
                classes.append('Adv')
            else:
                classes.append('OOV')
        return classes

    def verify_context(self, stems):
        for i, stem in enumerate(stems):
            if stem in self.subs:
                return True
        return False

    def analyze(self, sentence: str):
        # Verifica se a frase é gramaticalmente correta
        if verificar_gramatica(sentence):
            tokens, stems = self.tokenize_and_stem(sentence)

            contexto = self.verify_context(stems)

            if not contexto:
                return {'valid': False,
                        'error': 'Frase fora de contexto.',
                        'path': []}


            classes = self.classify(stems)
            state = self.dfa.initial_state

            stack = []
            path = [state]
            negacao_ativa = False 

            negacoes = {'não', 'nunca', 'jamais', 'nem'}

            for i, cls in enumerate(classes):
                token = tokens[i]
                # print(f"Token: {token}, Classe: {cls}, Estado: {state}")

                if token in negacoes:
                    negacao_ativa = True
                    continue 

                if cls == 'OOV':
                    return {'valid': False,
                            'error': f"Token '{token}' não reconhecido (OOV).",
                            'path': path}

                cls_original = cls

                if negacao_ativa:
                    if cls in ('AdjPos', 'AdjNeg', 'PosVerb'):
                        if cls == 'AdjPos' or cls == 'PosVerb':
                            cls = 'AdjNeg'
                        else:  # AdjNeg
                            cls = 'AdjPos'
                        negacao_ativa = False
                    else:
                        pass

                if cls == 'AdjPos' or cls == 'PosVerb':
                    stack.append('Pos')
                elif cls == 'AdjNeg':
                    if stack and stack[-1] == 'Pos':
                        stack.pop()
                    else:
                        stack.append('Neg')

                if cls not in self.dfa.transitions[state]:
                    expected = list(self.dfa.transitions[state].keys())
                    return {'valid': False,
                            'error': (f"Posição {i}: token '{token}' "
                                    f"(classe {cls_original}) não esperado em {state}; "
                                    f"esperava {expected}."),
                            'path': path}
                state = self.dfa.transitions[state][cls]
                path.append(state)

            if state in self.dfa.final_states:
                pos, neg = stack.count('Pos'), stack.count('Neg')
                mood = 'positivo' if pos > neg else 'negativo' if neg > pos else 'neutro'
                return {'valid': True, 'mood': mood, 'path': path}

            return {'valid': False,
                    'error': 'Frase terminou sem alcançar estado final de sentimento.',
                    'path': path}
        else:
            return {'valid': False,
                    'error': 'Frase não é gramaticalmente correta.',
                    'path': []}

    def draw_matplotlib(self, path=None, fname=None, seed=42, show=True):
        """
        Desenha o autômato com networkx e destaca o caminho percorrido.
        • path  – lista de estados devolvida por analyze()
        • fname – se fornecido, salva PNG/SVG; usa extensão do nome
        • show  – exibe na tela (útil em notebooks)
        """
        G = nx.DiGraph()

        # nós
        for st in self.dfa.states:
            G.add_node(st)

        # arestas com rótulo
        edge_labels = {}
        for src, trans in self.dfa.transitions.items():
            for sym, dst in trans.items():
                G.add_edge(src, dst)
                # vários símbolos podem ir para o mesmo dst; junte‐os
                edge_labels.setdefault((src, dst), []).append(sym)

        # layout consistente
        pos = nx.spring_layout(G, seed=seed)

        plt.figure(figsize=(9, 6))
        # nós
        nx.draw_networkx_nodes(G, pos,
                               nodelist=[st for st in G.nodes if st not in self.dfa.final_states],
                               node_size=1800, node_color='lightblue')
        nx.draw_networkx_nodes(G, pos,
                               nodelist=list(self.dfa.final_states),
                               node_size=1800, node_color='lightgreen')
        # todas as arestas em cinza
        nx.draw_networkx_edges(G, pos, width=1.2, alpha=.4)

        # se houver caminho, destaque‐o
        if path and len(path) > 1:
            highlight = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            nx.draw_networkx_edges(G, pos, edgelist=highlight,
                                   width=2.8, edge_color='red')

        # rótulos nos nós
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

        # rótulos nas arestas
        nice_labels = {k: ','.join(v) for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nice_labels, font_size=8)

        plt.title('Autômato – caminho percorrido', fontsize=14)
        plt.axis('off')
        plt.tight_layout()

        if fname:
            plt.savefig(fname, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
