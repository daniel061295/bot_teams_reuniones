# 🎙️ Teams Meeting Bot: Grabador y Resumidor con IA

Un sistema robusto diseñado para capturar reuniones de Microsoft Teams (o cualquier audio del sistema), transcribirlas localmente con Whisper y generar minutas ejecutivas estructuradas usando Google Gemini LLM.

Construido siguiendo los principios de **Arquitectura Limpia (Clean Architecture)** y **SOLID** para máxima mantenibilidad y extensibilidad.

## ✨ Características Principales

- **Captura Dual WASAPI Loopback**: Graba simultáneamente lo que escuchas por tus audíonos (otras personas) y lo que dices por el micrófono.
- **Transcripción Local con Whisper**: Privacidad total y sin costos de API para la transcripción (GPU/CUDA compatible).
- **VAD (Voice Activity Detection)**: Algoritmo basado en energía RMS que filtra silencios y ruidos antes de la transcripción, eliminando alucinaciones del modelo.
- **Resúmenes Inteligentes**: Generación automática de Decisiones Técnicas, Bloqueantes y Action Items usando la familia de modelos Gemini Flash.
- **Resiliencia de Modelos**: Sistema de fallback automático entre modelos de Gemini (`2.5-flash`, `2.0-flash`).
- **Arquitectura Premium**: Desacoplamiento total entre la lógica de negocio (Casos de Uso) y la infraestructura (Adapters).

## 🚀 Requisitos Previos

- **Python 3.14.2** (Recomendado)
- **FFmpeg**: Gestionado automáticamente por el script vía `imageio-ffmpeg` (sin instalación manual necesaria).
- **Windows 10/11**: Requerido para el soporte de WASAPI Loopback.

## 🛠️ Instalación

1. **Clonar el repositorio:**
   ```powershell
   git clone https://github.com/daniel061295/bot_teams_reuniones.git
   cd bot_teams_reuniones
   ```

2. **Crear y activar entorno virtual:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Instalar dependencias:**
   ```powershell
   pip install -r requirements.txt
   ```

## ⚙️ Configuración

Crea un archivo `.env` en la raíz del proyecto basado en `.env.example`:

```env
# Google Gemini API Key
GEMINI_API_KEY=tu_api_key_aqui

# Whisper Model (tiny, base, small, medium, large)
WHISPER_MODEL=medium

# Directorios de salida
OUTPUT_DIR=outputs
MINUTAS_DIR=minutas
```

## 📖 Uso

### Listar dispositivos de audio
Verifica qué audíonos o micrófonos están disponibles y detecta automáticamente el dispositivo predeterminado para el loopback:
```powershell
python main.py --list-devices
```

### Iniciar grabación de reunión
El bot empezará a grabar audio inmediatamente. Presiona `Ctrl+C` para detener la grabación e iniciar automáticamente la transcripción y el resumen.
```powershell
python main.py --id "Daily_Sync"
```

### Especificar micrófono
Si tienes múltiples micrófonos y quieres usar uno específico:
```powershell
python main.py --id "Arquitectura_Review" --mic "Nombre de tu Microfono"
```

## 🏗️ Estructura del Proyecto

```text
src/
├── domain/         # Entidades de negocio y Puertos (Interfaces)
├── use_cases/      # Lógica de aplicación (Record, Transcribe, Summarize)
├── infrastructure/ # Adaptadores (Audio, Whisper, Gemini, Persistence)
└── di/             # Inyección de dependencias con 'inject'
main.py             # Entrada CLI y orquestación
```

## 🧪 Pruebas
El proyecto incluye un set de pruebas unitarias para cada capa (26 tests, 91% cobertura):
```powershell
pytest
# Con reporte de cobertura:
pytest --cov=src --cov-report=term-missing
```

---
Desarrollado con ❤️ para optimizar la gestión de conocimiento en equipos técnicos.
