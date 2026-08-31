import os
import tarfile

def extract_text_from_latex_tarball(tar_path: str) -> str:
    """
    Estrae il testo esclusivamente dai file .tex presenti in un archivio tar.gz di arXiv.
    """
    if not os.path.exists(tar_path):
        print(f"[!] File not found: {tar_path}")
        return ""

    tex_content = []
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".tex"):
                    f = tar.extractfile(member)
                    if f:
                        raw_bytes = f.read()
                        try:
                            content = raw_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            content = raw_bytes.decode('latin-1')
                        tex_content.append(f"--- SOURCE FILE: {member.name} ---\n{content}\n")
    except Exception as e:
        print(f"[!] Error processing {tar_path}: {e}")

    return "\n".join(tex_content)
