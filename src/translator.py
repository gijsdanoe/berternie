# BERT and ERNIE Framework
# Copyright (C) 2024 Gijs Danoe
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from transformers import MarianMTModel, MarianTokenizer
import config


class Translator:
    def __init__(self, model_path=config.opus):
        self.tokenizer = MarianTokenizer.from_pretrained(model_path)
        self.model = MarianMTModel.from_pretrained(model_path)

    def translate(self,text):
        inputs = self.tokenizer(text, return_tensors='pt',truncation=True)
        translated_tokens = self.model.generate(**inputs, max_length=10)
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text