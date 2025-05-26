import json

#Clase auxiliar para que el json dump no deje los patterns tan feos

class CompactRowsEncoder(json.JSONEncoder):
    def encode(self, o):
        def encode_obj(obj):
            if isinstance(obj, dict):
                # For dicts, encode keys and values
                items = []
                for k, v in obj.items():
                    if k == "state" and isinstance(v, list) and all(isinstance(row, list) for row in v):
                        # Special case: encode 2D array compactly
                        compact_rows = '[' + ',\n'.join(
                            '[' + ','.join(str(x) for x in row) + ']' for row in v
                        ) + ']'
                        items.append(json.dumps(k) + ': ' + compact_rows)
                    else:
                        items.append(json.dumps(k) + ': ' + encode_obj(v))
                return '{' + ', '.join(items) + '}'
            elif isinstance(obj, list):
                # For lists, encode each element
                return '[' + ', '.join(encode_obj(el) for el in obj) + ']'
            else:
                # Default encode for other types
                return json.dumps(obj)

        return encode_obj(o)

