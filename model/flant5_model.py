import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentica.model.imagebind_encoder import ImageBindEncoder
from sentica.model.projection_layer import MMInputProjector

class FlanT5WithMM(nn.Module):
    def __init__(self, model_name="google/flan-t5-xxl", device="cuda"):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.llm = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.encoder = ImageBindEncoder(device)
        self.projector = MMInputProjector(input_dim=1024, output_dim=self.llm.config.d_model).to(device)
        self.device = device
        
        for name, param in self.llm.named_parameters():
            param.requires_grad = False 

        for param in self.encoder.model.parameters():
            param.requires_grad = False

        for name, param in self.projector.named_parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels, mm_paths=None):
        if mm_paths is not None:
            mm_embeddings = self.encoder.encode_batch(mm_paths).to(self.device)  # [B, 1024]
            mm_embeddings = self.projector(mm_embeddings)                        # [B, d_model]
            input_embeds = self.llm.encoder.embed_tokens(input_ids).clone()
            input_embeds[:, 0, :] = mm_embeddings
            encoder_outputs = self.llm.encoder(inputs_embeds=input_embeds, attention_mask=attention_mask)
            outputs = self.llm(encoder_outputs=encoder_outputs, attention_mask=attention_mask, labels=labels)
        else:
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs