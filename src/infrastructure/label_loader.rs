use std::collections::HashMap;
use std::path::Path;

use crate::domain::error::DomainError;

pub fn load_labels(path: &Path) -> Result<HashMap<usize, String>, DomainError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| DomainError::Config(format!("read {}: {e}", path.display())))?;

    let mut labels = HashMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (idx_str, name) = line
            .split_once(',')
            .ok_or_else(|| DomainError::Config(format!("invalid label line: {line}")))?;
        let idx: usize = idx_str
            .trim()
            .parse()
            .map_err(|e| DomainError::Config(format!("invalid index '{idx_str}': {e}")))?;
        labels.insert(idx, name.trim().to_string());
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(content: &str) -> tempfile::NamedTempFile {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        write!(tmp, "{content}").unwrap();
        tmp
    }

    #[test]
    fn should_parse_index_and_name_pairs() {
        let tmp = write_temp("0,alpha\n1,beta\n");
        let labels = load_labels(tmp.path()).unwrap();

        assert_eq!(labels.len(), 2);
        assert_eq!(labels[&0], "alpha");
        assert_eq!(labels[&1], "beta");
    }

    #[test]
    fn should_skip_empty_lines() {
        let tmp = write_temp("0,alpha\n\n1,beta\n");
        let labels = load_labels(tmp.path()).unwrap();
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn should_trim_whitespace() {
        let tmp = write_temp("  0 , alpha  \n");
        let labels = load_labels(tmp.path()).unwrap();
        assert_eq!(labels[&0], "alpha");
    }

    #[test]
    fn should_return_error_for_missing_comma() {
        let tmp = write_temp("no_comma_here\n");
        assert!(load_labels(tmp.path()).is_err());
    }

    #[test]
    fn should_return_error_for_non_numeric_index() {
        let tmp = write_temp("abc,alpha\n");
        assert!(load_labels(tmp.path()).is_err());
    }

    #[test]
    fn should_return_error_for_missing_file() {
        assert!(load_labels(Path::new("/nonexistent/labels.csv")).is_err());
    }
}
