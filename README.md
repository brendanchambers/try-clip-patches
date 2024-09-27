# learning about CLIP in the context of image tokenization  

## understand image tokenization given a pretrained CLIP model  
  
e.g.  
image -> [processor] -> pixels [bsz x 3 x 224 x 224]  
pixels -> [patch embeddings] -> 3 x 1024 x 14
pixels -> [embeddings] -> token and pos embedding [bxz x 768]
embeddings -> [transformer encoder] -> final hidden layer [bsz x seq len x 1024]
