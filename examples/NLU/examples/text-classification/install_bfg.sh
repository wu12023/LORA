# --------------------
# 1. Download it
# --------------------

# Option 1 [easiest]: download the latest version
wget https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo

# OR Option 2: get the latest *released* version.
# Go here and find the latest release:
# https://github.com/newren/git-filter-repo/releases
# Use that release number in the cmds below.
wget https://github.com/newren/git-filter-repo/releases/download/v2.38.0/git-filter-repo-2.38.0.tar.xz
# install dependencies
sudo apt install xz-utils  
# extract the downloaded archive
tar -xf git-filter-repo-2.38.0.tar.xz
# copy out the executable; we'll move it to a directory within our PATH later
cp git-filter-repo-2.38.0/git-filter-repo .

# --------------------
# 2. make it executable
# --------------------
chmod +x git-filter-repo

# --------------------
# 3. move it to a dir in your PATH
# --------------------

# Option 1 [easiest]: make this executable accessible to ALL users
sudo mv -i git-filter-repo /usr/local/bin

# OR Option 2: make this executable accessible to your user only
mkdir -p ~/bin
mv -i git-filter-repo ~/bin 
# add ~/bin to your path by re-sourcing your `~/.profile` file.
# This works on Ubuntu if you are using the default ~/.profile file, which can
# also be found in /etc/skel/.profile, by the way.
. ~/.profile

# --------------------
# 4. Done. Now run it.
# --------------------
git filter-repo --version       # check the version
git filter-repo -h              # help menu
git filter-repo -h | less -RFX  # help menu, viewed in the `less` viewer

