#include <iostream>
#include <string>
#include <vector>
#include <map>

// Potencia modular para RSA
long long modExp(long long base, long long exp, long long mod) {
    long long result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) result = (result * base) % mod;
        base = (base * base) % mod;
        exp /= 2;
    }
    return result;
}

// Tabla de sustitución simétrica
char sustitucion(char c) {
    std::map<char,char> tabla = {
        {'a','z'},{'b','y'},{'c','x'},{'d','w'},{'e','v'},{'f','u'},
        {'g','t'},{'h','s'},{'i','r'},{'j','q'},{'k','p'},{'l','o'},
        {'m','n'},{'n','m'},{'o','l'},{'p','k'},{'q','j'},{'r','i'},
        {'s','h'},{'t','g'},{'u','f'},{'v','e'},{'w','d'},{'x','c'},
        {'y','b'},{'z','a'},
        {'1','0'},{'0','1'},{'2','9'},{'9','2'},{'3','8'},{'8','3'},
        {'4','7'},{'7','4'},{'5','6'},{'6','5'}
    };
    if(tabla.count(c)) return tabla[c];
    return c;
}

// Desencripción RSA: cada bloque de 4 dígitos
std::string desencriptarRSA(const std::string &cadena, long long clave, long long n) {
    std::string resultado;
    for (size_t i = 0; i < cadena.size(); i += 4) {
        std::string bloque = cadena.substr(i, 4);
        long long valor = std::stoll(bloque);
        long long descifrado = modExp(valor, clave, n);
        resultado += (char)descifrado; // convertir a caracter
    }
    return resultado;
}

// Aplicar sustitución simétrica a todo el texto
std::string aplicarSustitucion(const std::string &texto) {
    std::string claro;
    for (char c : texto) claro += sustitucion(c);
    return claro;
}

int main() {
    std::string cadena;
    std::cout << "Ingrese la cadena de bloques (cada 4 dígitos): ";
    std::getline(std::cin, cadena);
//LLAVES RSA; variales 
    long long privadaA = 17;
    long long privadaB = 593;
    long long publica = 1517;

    // Primera desencripción RSA con clave privada A
    std::string intermedio = desencriptarRSA(cadena, privadaA, publica);
    std::cout << "Primera desencripción RSA: " << intermedio << std::endl;

    // Segunda desencripción: aplicar sustitución simétrica
    std::string claro = aplicarSustitucion(intermedio);
    std::cout << "Mensaje final claro: " << claro << std::endl;

    return 0;
}