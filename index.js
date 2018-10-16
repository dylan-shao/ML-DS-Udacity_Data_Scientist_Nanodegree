/*
* Automatically create README.md at root with links to subfolders
*/

const fs = require('fs');

const fileName = 'README.md';
fs.writeFileSync(fileName, '');

let level = 1;

(function f(dir, level) {
  const filesOrFolders = fs.readdirSync(dir);
  for (let i = 0; i < filesOrFolders.length; i++) {
    const fileOrFolder = filesOrFolders[i];
    // ignore unwanted files or folders
    if (!/^(node_modules)|.git$|helpers$|.checkpoints$|^(tmp)|.json$|^index\.js.*$|ummary.md$|README.md$|((^|[\/\\])\..)/.test(fileOrFolder)) {
      const fileOrFolderName = dir + '/' + fileOrFolder;

      /*---------------------if is directory-------------------*/
      if (_isFolder(fileOrFolderName)) {
        // fileOrFolderName start with ./path/to...., so remove it
        const folderName = fileOrFolderName.replace('./', '');
        if(level === 1) {
          _append(`## ${folderName}\n\n`);
        }

        if(level === 2) {
          const folderPath = (encodeURIComponent(dir.replace('./', '')) + '/' + encodeURIComponent(fileOrFolder));
          const path = _getUrl(folderPath);
          _append(`[${fileOrFolder}](${path})\n\n`);
        }
        level === 1 && f(fileOrFolderName, level+1);
        /*-------------------------------------------------------*/
      }
    }
  }
})('.', level);

/*------------------------------Utilities-------------------------------------*/
function _isFolder(fileOrFolderName) {
  return fs.statSync(fileOrFolderName).isDirectory();
}

function _getUrl(path, username = 'dylan-shao', repoName = 'ML-DS-Udacity_Data_Scientist_Nanodegree', branchName = 'master') {
  return `https://github.com/${username}/${repoName}/blob/${branchName}/${path}`;
}

function _append(content) {
  fs.appendFile(fileName, content, function(err) {
    if (err) throw err;
  });
}
