# rewrite

interesting, so for this rewrite I'm focusing on idiomaticity both in regards to zig
but also the the domain I've been looking into more details the format safetensor
ans more specifically other implementation, I'll use this one as a reference
https://github.com/safetensors/safetensors/blob/main/safetensors/src/tensor.rs

note to self :

I'm sorting tensors exclusively by name whereas in the tensor.rs they sort by
Dtype aligment and them by name, i need to look into that

```rs
fn prepare<S, V, I>(
    data: I,
    data_info: Option<HashMap<String, String>>,
) -> Result<(PreparedData, Vec<V>), SafeTensorError>
where
    S: AsRef<str> + Ord + Display,
    V: View,
    I: IntoIterator<Item = (S, V)>,
{
    // Make sure we're sorting by descending dtype alignment
    // Then by name
    let mut data: Vec<_> = data.into_iter().collect();
    data.sort_by(|(lname, left), (rname, right)| {
        right.dtype().cmp(&left.dtype()).then(lname.cmp(rname))
    });

    let mut tensors: Vec<V> = Vec::with_capacity(data.len());
    let mut hmetadata = Vec::with_capacity(data.len());
    let mut offset = 0;

    for (name, tensor) in data {
        let n = tensor.data_len();
        let tensor_info = TensorInfo {
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
            data_offsets: (offset, offset + n),
        };
        offset += n;
        hmetadata.push((name.to_string(), tensor_info));
        tensors.push(tensor);
    }

    let metadata: Metadata = Metadata::new(data_info, hmetadata)?;
    let mut metadata_buf = serde_json::to_string(&metadata)?.into_bytes();

    // Force alignment to 8 bytes.
    let aligned_metadata_len = metadata_buf.len().next_multiple_of(N_LEN);
    metadata_buf.resize(aligned_metadata_len, b' ');

    Ok((
        PreparedData {
            n: aligned_metadata_len as u64,
            header_bytes: metadata_buf,
            offset,
        },
        tensors,
    ))
}
```

they also use direct Io i need to investigate wether Zig does that in the Io 
interface (or if i need to do it)

```rs
#[cfg(feature = "std")]
fn buffered_write_to_file<V: View>(
    path: impl AsRef<Path>,
    n: u64,
    header_bytes: &[u8],
    tensors: &[V],
    total_size: usize,
) -> Result<(), SafeTensorError> {
    let path = path.as_ref();
    // Write to a sibling tempfile then rename, so an existing `path` is never
    // truncated under any mmap of it (e.g. tensors returned by `load_file`).
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let temp = tempfile::NamedTempFile::new_in(parent)?;

    temp.as_file().set_len(total_size as u64)?;

    // Serialize tensors to a file using direct I/O (bypassing page cache) using F_NOCACHE.
    // This yields ~30% performance improvement.
    #[cfg(target_os = "macos")]
    unsafe {
        use std::os::fd::AsRawFd;

        libc::fcntl(temp.as_file().as_raw_fd(), libc::F_NOCACHE, 1);
    }

    {
        let mut f = std::io::BufWriter::with_capacity(1024 * 1024, temp.as_file());

        f.write_all(n.to_le_bytes().as_ref())?;
        f.write_all(header_bytes)?;

        for tensor in tensors {
            f.write_all(tensor.data().as_ref())?;
        }

        f.flush()?;
    }

    temp.persist(path).map_err(|e| e.error)?;

    Ok(())
}
```

they use the same strategy as me to reduce memory usage whitch is to stream
the decompressed output.

they also use different types, then me for tensors, I have to dig a bit deeper
it seems that their approach is to lazily load tensors, which mine doesn't do
i guess because mine is optimized for a single file not for potentially a
distributed system.

they have a lot of tests so if I adapt them the tensor parsing should be solid.


trying to also collect all the sources I've used to help me with the impl

https://github.com/ingted/nvfp4_native/blob/master/libNVFP4.cpp
https://github.com/r-chong/mxfp4-dequantizer
https://github.com/foundation-model-stack/fastsafetensors


found also a json schema for the safetensors format
https://github.com/safetensors/safetensors/blob/main/docs/safetensors.schema.json

ok I've gathered enough sources, I'll start designing a better architecture


so let start designing a better system from first princple. The exercise is 

input  = 1 .safetensors NVFP4
output = 1 .safetensors dequantized weighs

a safetensor file is composed of 3 main parts

[0........8][0.......................headersize][sizeof(u64) + headersize.......EOF]
[headersize][json header representing the model][binary blobs indexed by the header]

as per the safetensors repo :


    8 bytes: N, an unsigned little-endian 64-bit integer, containing the size of the header
    N bytes: a JSON UTF-8 string representing the header.
        The header data MUST begin with a { character (0x7B).
        The header data MAY be trailing padded with whitespace (0x20).
        The header is a dict like {"TENSOR_NAME": {"dtype": "F16", "shape": [1, 16, 256], "data_offsets": [BEGIN, END]}, "NEXT_TENSOR_NAME": {...}, ...},
            data_offsets point to the tensor data relative to the beginning of the byte buffer (i.e. not an absolute position in the file), with BEGIN as the starting offset and END as the one-past offset (so total tensor byte size = END - BEGIN).
        A special key __metadata__ is allowed to contain free form string-to-string map. Arbitrary JSON is not allowed, all values must be strings.
    Rest of the file: byte-buffer.

Notes:

    Duplicate keys are disallowed. Not all parsers may respect this.
    In general the subset of JSON is implicitly decided by serde_json for this library. Anything obscure might be modified at a later time, that odd ways to represent integer, newlines and escapes in utf-8 strings. This would only be done for safety concerns
    Tensor values are not checked against, in particular NaN and +/-Inf could be in the file
    Empty tensors (tensors with 1 dimension being 0) are allowed. They are not storing any data in the databuffer, yet retaining size in the header. They don't really bring a lot of values but are accepted since they are valid tensors from traditional tensor libraries perspective (torch, tensorflow, numpy, ..).
    0-rank Tensors (tensors with shape []) are allowed, they are merely a scalar.
    The byte buffer needs to be entirely indexed, and cannot contain holes. This prevents the creation of polyglot files.
    Endianness: Little-endian. moment.
    Order: 'C' or row-major.
    Notes: Some smaller than 1 byte dtypes appeared, which make alignment tricky. Non traditional APIs might be required for those.

