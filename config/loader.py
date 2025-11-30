# config/loader.py
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class Config:
  raw: Dict[str, Any]

  @classmethod
  def from_yaml(cls, path: str) -> "Config":
      with open(path, "r") as f:
          data = yaml.safe_load(f)
      return cls(raw=data)

  def __getitem__(self, key: str) -> Any:
      return self.raw[key]

  def get(self, key: str, default=None) -> Any:
      return self.raw.get(key, default)
