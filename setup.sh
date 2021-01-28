mkdir -p ~/.streamlit/
apt install ffmpeg
echo "[general]
email = \"anshulchoudhary2001@gmail.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
