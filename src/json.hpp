#ifndef NW_SRC_JSON_HPP
#define NW_SRC_JSON_HPP

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <iostream>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace nekwave {

enum class JsonType {
    Null,
    Boolean,
    Number,
    String,
    Array,
    Object
};

class JsonValue {
public:
    JsonType type;
    bool boolVal;
    double numVal;
    std::string strVal;
    std::vector<JsonValue> arrVal;
    std::map<std::string, JsonValue> objVal; // Ordered map for predictable traversal

    JsonValue() : type(JsonType::Null), boolVal(false), numVal(0.0) {}
    explicit JsonValue(bool b) : type(JsonType::Boolean), boolVal(b), numVal(b ? 1.0 : 0.0) {}
    explicit JsonValue(double n) : type(JsonType::Number), boolVal(n != 0.0), numVal(n) {}
    explicit JsonValue(int n) : type(JsonType::Number), boolVal(n != 0), numVal(static_cast<double>(n)) {}
    explicit JsonValue(long n) : type(JsonType::Number), boolVal(n != 0), numVal(static_cast<double>(n)) {}
    explicit JsonValue(long long n) : type(JsonType::Number), boolVal(n != 0), numVal(static_cast<double>(n)) {}
    explicit JsonValue(const std::string& s) : type(JsonType::String), boolVal(!s.empty()), numVal(0.0), strVal(s) {}
    explicit JsonValue(const char* s) : type(JsonType::String), boolVal(s && s[0] != '\0'), numVal(0.0), strVal(s ? s : "") {}
    explicit JsonValue(JsonType t) : type(t), boolVal(false), numVal(0.0) {}

    bool isNull() const { return type == JsonType::Null; }
    bool isBool() const { return type == JsonType::Boolean; }
    bool isNumber() const { return type == JsonType::Number; }
    bool isString() const { return type == JsonType::String; }
    bool isArray() const { return type == JsonType::Array; }
    bool isObject() const { return type == JsonType::Object; }

    std::string asString(const std::string& def = "") const {
        if (type == JsonType::String) return strVal;
        if (type == JsonType::Number) {
            std::ostringstream ss;
            if (std::floor(numVal) == numVal && !std::isinf(numVal)) {
                ss << static_cast<long long>(numVal);
            } else {
                ss << numVal;
            }
            return ss.str();
        }
        if (type == JsonType::Boolean) return boolVal ? "true" : "false";
        if (type == JsonType::Null) return def;
        return def;
    }

    double asDouble(double def = 0.0) const {
        if (type == JsonType::Number) return numVal;
        if (type == JsonType::String) {
            try { return std::stod(strVal); } catch (...) { return def; }
        }
        if (type == JsonType::Boolean) return boolVal ? 1.0 : 0.0;
        return def;
    }

    int asInt(int def = 0) const {
        if (type == JsonType::Number) return static_cast<int>(numVal);
        if (type == JsonType::String) {
            try { return std::stoi(strVal); } catch (...) { return def; }
        }
        if (type == JsonType::Boolean) return boolVal ? 1 : 0;
        return def;
    }

    bool asBool(bool def = false) const {
        if (type == JsonType::Boolean) return boolVal;
        if (type == JsonType::Number) return numVal != 0.0;
        if (type == JsonType::String) {
            std::string s = strVal;
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
            if (s == "true" || s == "1" || s == "yes" || s == "on") return true;
            if (s == "false" || s == "0" || s == "no" || s == "off") return false;
        }
        return def;
    }

    size_t size() const {
        if (type == JsonType::Array) return arrVal.size();
        if (type == JsonType::Object) return objVal.size();
        return 0;
    }

    bool has(const std::string& key) const {
        if (type != JsonType::Object) return false;
        return objVal.find(key) != objVal.end();
    }

    const JsonValue& operator[](const std::string& key) const {
        static const JsonValue nullVal;
        if (type != JsonType::Object) return nullVal;
        auto it = objVal.find(key);
        return (it != objVal.end()) ? it->second : nullVal;
    }

    JsonValue& operator[](const std::string& key) {
        if (type != JsonType::Object) {
            type = JsonType::Object;
            objVal.clear();
        }
        return objVal[key];
    }

    const JsonValue& operator[](size_t index) const {
        static const JsonValue nullVal;
        if (type != JsonType::Array || index >= arrVal.size()) return nullVal;
        return arrVal[index];
    }

    JsonValue& operator[](size_t index) {
        if (type != JsonType::Array) {
            type = JsonType::Array;
            arrVal.clear();
        }
        if (index >= arrVal.size()) {
            arrVal.resize(index + 1);
        }
        return arrVal[index];
    }

    std::vector<std::string> keys() const {
        std::vector<std::string> res;
        if (type == JsonType::Object) {
            for (const auto& kv : objVal) {
                res.push_back(kv.first);
            }
        }
        return res;
    }

    static JsonValue parse(const std::string& jsonStr, std::string* errorMsg = nullptr) {
        Parser parser(jsonStr);
        return parser.parse(errorMsg);
    }

private:
    class Parser {
    public:
        explicit Parser(const std::string& s) : src(s), pos(0), len(s.length()) {}

        JsonValue parse(std::string* errorMsg) {
            try {
                skipWhitespaceAndComments();
                if (pos >= len) {
                    if (errorMsg) *errorMsg = "Empty input";
                    return JsonValue();
                }
                JsonValue res = parseValue();
                skipWhitespaceAndComments();
                return res;
            } catch (const std::exception& e) {
                if (errorMsg) *errorMsg = e.what();
                return JsonValue();
            }
        }

    private:
        const std::string& src;
        size_t pos;
        size_t len;

        void skipWhitespaceAndComments() {
            while (pos < len) {
                char c = src[pos];
                if (std::isspace(static_cast<unsigned char>(c))) {
                    pos++;
                    continue;
                }
                // Handle // single line comments
                if (c == '/' && pos + 1 < len && src[pos + 1] == '/') {
                    pos += 2;
                    while (pos < len && src[pos] != '\n' && src[pos] != '\r') {
                        pos++;
                    }
                    continue;
                }
                // Handle /* block comments */
                if (c == '/' && pos + 1 < len && src[pos + 1] == '*') {
                    pos += 2;
                    while (pos + 1 < len && !(src[pos] == '*' && src[pos + 1] == '/')) {
                        pos++;
                    }
                    if (pos + 1 < len) pos += 2;
                    continue;
                }
                // Handle # single line comments (common in scientific script config files)
                if (c == '#') {
                    pos++;
                    while (pos < len && src[pos] != '\n' && src[pos] != '\r') {
                        pos++;
                    }
                    continue;
                }
                break;
            }
        }

        JsonValue parseValue() {
            skipWhitespaceAndComments();
            if (pos >= len) throw std::runtime_error("Unexpected end of input");

            char c = src[pos];
            if (c == '{') return parseObject();
            if (c == '[') return parseArray();
            if (c == '"' || c == '\'') return parseString();
            if (c == 't' || c == 'T' || c == 'f' || c == 'F') return parseBoolean();
            if (c == 'n' || c == 'N') return parseNull();
            if (c == '-' || c == '+' || std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
                return parseNumber();
            }

            // Word / identifier fallback (e.g. unquoted auto, true, none, box)
            return parseIdentifier();
        }

        JsonValue parseObject() {
            JsonValue val(JsonType::Object);
            pos++; // Skip '{'

            while (true) {
                skipWhitespaceAndComments();
                if (pos >= len) throw std::runtime_error("Unterminated object: missing '}'");
                if (src[pos] == '}') {
                    pos++;
                    break;
                }

                // Parse key
                std::string key;
                if (src[pos] == '"' || src[pos] == '\'') {
                    key = parseRawString();
                } else {
                    key = parseRawIdentifier();
                }

                skipWhitespaceAndComments();
                if (pos >= len || (src[pos] != ':' && src[pos] != '=')) {
                    throw std::runtime_error("Expected ':' or '=' after key '" + key + "'");
                }
                pos++; // Skip ':' or '='

                skipWhitespaceAndComments();
                JsonValue child = parseValue();
                val.objVal[key] = child;

                skipWhitespaceAndComments();
                if (pos < len && src[pos] == ',') {
                    pos++; // Skip ','
                } else if (pos < len && src[pos] == '}') {
                    pos++;
                    break;
                } else if (pos < len) {
                    // Tolerant to newline separation without comma
                    continue;
                } else {
                    break;
                }
            }
            return val;
        }

        JsonValue parseArray() {
            JsonValue val(JsonType::Array);
            pos++; // Skip '['

            while (true) {
                skipWhitespaceAndComments();
                if (pos >= len) throw std::runtime_error("Unterminated array: missing ']'");
                if (src[pos] == ']') {
                    pos++;
                    break;
                }

                val.arrVal.push_back(parseValue());

                skipWhitespaceAndComments();
                if (pos < len && src[pos] == ',') {
                    pos++; // Skip ','
                } else if (pos < len && src[pos] == ']') {
                    pos++;
                    break;
                } else if (pos < len) {
                    // Tolerant to newline separation
                    continue;
                } else {
                    break;
                }
            }
            return val;
        }

        std::string parseRawString() {
            char quote = src[pos++];
            std::string res;
            while (pos < len) {
                char c = src[pos++];
                if (c == quote) {
                    return res;
                }
                if (c == '\\' && pos < len) {
                    char esc = src[pos++];
                    if (esc == '"') res += '"';
                    else if (esc == '\'') res += '\'';
                    else if (esc == '\\') res += '\\';
                    else if (esc == '/') res += '/';
                    else if (esc == 'b') res += '\b';
                    else if (esc == 'f') res += '\f';
                    else if (esc == 'n') res += '\n';
                    else if (esc == 'r') res += '\r';
                    else if (esc == 't') res += '\t';
                    else res += esc;
                } else {
                    res += c;
                }
            }
            return res;
        }

        JsonValue parseString() {
            return JsonValue(parseRawString());
        }

        std::string parseRawIdentifier() {
            size_t start = pos;
            while (pos < len) {
                char c = src[pos];
                if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.') {
                    pos++;
                } else {
                    break;
                }
            }
            return src.substr(start, pos - start);
        }

        JsonValue parseIdentifier() {
            std::string id = parseRawIdentifier();
            std::string lid = id;
            std::transform(lid.begin(), lid.end(), lid.begin(), [](unsigned char c) { return std::tolower(c); });
            if (lid == "true") return JsonValue(true);
            if (lid == "false") return JsonValue(false);
            if (lid == "null") return JsonValue(JsonType::Null);
            return JsonValue(id);
        }

        JsonValue parseBoolean() {
            return parseIdentifier();
        }

        JsonValue parseNull() {
            return parseIdentifier();
        }

        JsonValue parseNumber() {
            size_t start = pos;
            if (pos < len && (src[pos] == '-' || src[pos] == '+')) pos++;
            while (pos < len && std::isdigit(static_cast<unsigned char>(src[pos]))) pos++;
            if (pos < len && src[pos] == '.') {
                pos++;
                while (pos < len && std::isdigit(static_cast<unsigned char>(src[pos]))) pos++;
            }
            if (pos < len && (src[pos] == 'e' || src[pos] == 'E')) {
                pos++;
                if (pos < len && (src[pos] == '-' || src[pos] == '+')) pos++;
                while (pos < len && std::isdigit(static_cast<unsigned char>(src[pos]))) pos++;
            }
            std::string numStr = src.substr(start, pos - start);
            try {
                double val = std::stod(numStr);
                return JsonValue(val);
            } catch (...) {
                return JsonValue(numStr);
            }
        }
    };
};

} // namespace nekwave

#endif // NW_SRC_JSON_HPP
