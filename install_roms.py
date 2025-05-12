from ale_py.roms import install_roms

def install():
    try:
        install_roms()
        print("ROMs instalados com sucesso!")
    except Exception as e:
        print(f"Erro ao instalar ROMs: {str(e)}")

if __name__ == "__main__":
    install()
