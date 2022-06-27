# download images

function download_data() {
    url="https://labmgf.dica.polimi.it/pujob/francesco-git/"
    fname="img.tar.gz"
    wget --no-check-certificate --show-progress "$url$fname"
    tar -xzvf "$fname"
    rm "$fname"
    cd ../
}

download_data;

