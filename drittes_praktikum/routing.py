from graphviz import Digraph

# Create a UML component diagram using Graphviz
uml = Digraph(format='pdf', name='UML_Component_Diagram')
uml.attr(rankdir='LR', size='10')

# Define components
uml.node('Benutzerverwaltung', label='<<Component>>\nBenutzerverwaltung')
uml.node('DeckVerwaltung', label='<<Component>>\nDeck-Verwaltung')
uml.node('KartenManagement', label='<<Component>>\nKarten-Management')
uml.node('General', label='<<Component>>\nGeneral')

# Define dependencies
uml.edge('Benutzerverwaltung', 'DeckVerwaltung', label='depends on')
uml.edge('DeckVerwaltung', 'KartenManagement', label='depends on')
uml.edge('Benutzerverwaltung', 'General', label='uses')
uml.edge('DeckVerwaltung', 'General', label='uses')
uml.edge('KartenManagement', 'General', label='uses')

# Render the diagram to a PDF
output_path = '/mnt/data/UML_Component_Diagram.pdf'
uml.render(output_path, cleanup=True)
output_path
