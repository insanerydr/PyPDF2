import camelot
import pyPdf2 as pyPdf
from tabula import read_pdf
from matplotlib.pyplot import plt
tables = camelot.read_pdf('invoice.pdf' ,flavor='stream')
camelot.plot(tables[0], kind='text')
plt.show()
camelot.plot(tables[0], kind='grid')
plt.show()

reader = pyPdf.PdfFileReader(open("C:\Users\riley\Desktop\Bank Statements\50340.pdf", mode='rb' ))
n = reader.getNumPages() 

df = []
for page in [str(i+1) for i in range(n)]:
    if page == "1":
            df.append(read_pdf(r"C:\Users\riley\Desktop\Bank Statements\50340.pdf", area=(530,12.75,790.5,561), pages=page))
    else:
            df.append(read_pdf(r"C:\Users\riley\Desktop\Bank Statements\50340.pdf", pages=page))

import tensorflow as tf
import tensorflow_datasets
from transformers import *

# Load dataset, tokenizer, model from pretrained model/vocabulary
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased')

# If you used to have this line in pytorch-pretrained-bert:
loss = model(input_ids, labels=labels)

# Now just use this line in transformers to extract the loss from the output tuple:
outputs = model(input_ids, labels=labels)
loss = outputs[0]

# In transformers you can also have access to the logits:
loss, logits = outputs[:2]

# And even the attention weights if you configure the model to output them (and other outputs too, see the docstrings and documentation)
model = BertForTokenClassification.from_pretrained('bert-base-uncased', output_attentions=True)
outputs = model(input_ids, labels=labels)
loss, logits, attentions = outputs

