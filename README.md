# 📖 Directorio del Proyecto
Este repositorio contiene todo lo que se usará durante el desarrollo del sistema móvil de procesamiento de lenguaje de señas.

## 💾 Clonar Repositorio
Para comenzar con la clonación del repositorio se sugiere usar este comando en terminal, esto descargará de manera local el repositorio junto con los submódulos de desarrollo

```bash
git clone --recurse-submodules https://github.com/Asaph-Velazquez/ASL-System.git
```

para dirigirte a cualquiera de los submódulos de desarrollo usar

```bash
cd "Nombre del modulo de desarrollo"
```

## 🗃️ Estructura del proyecto (Submódulos de desarrollo)
Durante el desarrollo del proyecto se trabajará con diferentes módulos los cuales se encuentran organizados de la siguiente forma:

- `ASL-IA`: Esta carpeta contiene el directorio del desarrollo del modelo de Machine Learning, principalmente desarrollado en Python.

- `ASL-MobileApp`: Esta carpeta contiene el directorio de desarrollo de la aplicación móvil, desarrollada principalmente con Expo.dev + TypeScript.

- `ASL-Web`: Esta carpeta contiene el directorio de desarrollo del panel web, desarrollado con React + Vite + TypeScript. Su función es recibir y visualizar las peticiones de la aplicación móvil.