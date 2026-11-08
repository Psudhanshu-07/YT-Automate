# Background music (optional)

Drop **one** royalty-free audio file in this folder and the pipeline will mix it
under the voiceover at low volume (ducked with a sidechain compressor).

* Supported formats: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`
* If several files are present, the alphabetically first one is used unless you
  set `music.track` in `config.yaml`.
* If the folder is empty, the video is rendered with voiceover only. The run
  does **not** fail.

## Only use audio you are allowed to publish

Free options (keep a note of the licence and attribution requirements):

| Source | Notes |
| --- | --- |
| [YouTube Audio Library](https://studio.youtube.com/channel/UC/music) | Filter to "no attribution required" |
| [Pixabay Music](https://pixabay.com/music/) | Pixabay Content License |
| [Free Music Archive](https://freemusicarchive.org/) | Check each track's licence |
| [Incompetech / Kevin MacLeod](https://incompetech.com/music/royalty-free/) | CC BY - attribution required |

Do **not** commit copyrighted music to this repository. Anything you commit here
is published publicly under this repo's licence.

Recommended: an instrumental, 60-120 BPM, no vocals (vocals fight the
voiceover), and at least as long as your video (the track is trimmed and faded
automatically).
