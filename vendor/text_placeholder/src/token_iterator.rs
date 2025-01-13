enum State {
    Text,
    Placeholder,
}

#[derive(Clone, PartialEq, Debug)]
pub enum Token<'t> {
    Text(&'t str),
    Placeholder(&'t str),
}

pub struct TokenIterator<'t> {
    text: &'t str,
    state: State,
    start: &'t str,
    end: &'t str,
}

impl<'t> TokenIterator<'t> {
    pub fn new(text: &'t str, start: &'t str, end: &'t str) -> Self {
        Self {
            text,
            start,
            end,
            state: State::Text,
        }
    }

    fn parse_text(&mut self) -> Token<'t> {
        let token: Token;

        if let Some(placeholder_index) = self.text.find(self.start) {
            token = Token::Text(&self.text[..placeholder_index]);
            self.text = &self.text[placeholder_index..];
            self.state = State::Placeholder;
        } else {
            token = Token::Text(self.text);
            self.text = "";
        }

        token
    }

    fn parse_placeholder(&mut self) -> Token<'t> {
        let token: Token;
        self.state = State::Text;

        if let Some(placeholder_index) = self.text.find(self.end) {
            token = Token::Placeholder(
                self.text[self.start.len()..placeholder_index]
                    .trim_start_matches(' ')
                    .trim_end_matches(' '),
            );
            let new_position = placeholder_index + self.end.len();
            self.text = &self.text[new_position..];
        } else {
            token = Token::Text(self.text);
            self.text = "";
        }

        token
    }
}

impl<'t> Iterator for TokenIterator<'t> {
    type Item = Token<'t>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.text.is_empty() {
            return None;
        }

        match self.state {
            State::Text => Some(self.parse_text()),
            State::Placeholder => Some(self.parse_placeholder()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Token, TokenIterator};
    extern crate alloc;
    use alloc::vec::Vec;

    #[test]
    fn test_no_boundaries_present() {
        let tokens: Vec<Token> = TokenIterator::new("hello world", "[", "]").collect();
        assert_eq!(tokens, vec![Token::Text("hello world")]);
    }

    #[test]
    fn test_boundary_begging_of_line() {
        let tokens: Vec<Token> = TokenIterator::new("[placeholder] text", "[", "]").collect();
        assert_eq!(
            tokens,
            vec![
                Token::Text(""),
                Token::Placeholder("placeholder"),
                Token::Text(" text")
            ]
        );
    }

    #[test]
    fn test_boundary_middle_of_line() {
        let tokens: Vec<Token> = TokenIterator::new("text [placeholder] text", "[", "]").collect();
        assert_eq!(
            tokens,
            vec![
                Token::Text("text "),
                Token::Placeholder("placeholder"),
                Token::Text(" text")
            ]
        );
    }

    #[test]
    fn test_boundary_end_of_line() {
        let tokens: Vec<Token> = TokenIterator::new("text [placeholder]", "[", "]").collect();
        assert_eq!(
            tokens,
            vec![Token::Text("text "), Token::Placeholder("placeholder")]
        );
    }

    #[test]
    fn test_boundary_multiple_boundaries() {
        let tokens: Vec<Token> = TokenIterator::new(
            "[placeholder] text [placeholder] test [placeholder]",
            "[",
            "]",
        )
        .collect();
        assert_eq!(
            tokens,
            vec![
                Token::Text(""),
                Token::Placeholder("placeholder"),
                Token::Text(" text "),
                Token::Placeholder("placeholder"),
                Token::Text(" test "),
                Token::Placeholder("placeholder")
            ]
        );
    }

    #[test]
    fn test_missing_boundary_start() {
        let tokens: Vec<Token> = TokenIterator::new("text placeholder]", "[", "]").collect();
        assert_eq!(tokens, vec![Token::Text("text placeholder]")]);
    }

    #[test]
    fn test_missing_boundary_end() {
        let tokens: Vec<Token> = TokenIterator::new("text [placeholder", "[", "]").collect();
        assert_eq!(
            tokens,
            vec![Token::Text("text "), Token::Text("[placeholder")]
        );
    }

    #[test]
    fn test_boundary_and_missing_boundaries() {
        let tokens: Vec<Token> =
            TokenIterator::new("text [placeholder] [placeholder", "[", "]").collect();
        assert_eq!(
            tokens,
            vec![
                Token::Text("text "),
                Token::Placeholder("placeholder"),
                Token::Text(" "),
                Token::Text("[placeholder")
            ]
        );
    }

    #[test]
    fn test_multiple_chars_boundary_start_of_line() {
        let tokens: Vec<Token> = TokenIterator::new("{{placeholder}} text", "{{", "}}").collect();
        assert_eq!(
            tokens,
            vec![
                Token::Text(""),
                Token::Placeholder("placeholder"),
                Token::Text(" text")
            ]
        );
    }

    #[test]
    fn test_multiple_chars_boundary_middle_of_line() {
        let tokens: Vec<Token> =
            TokenIterator::new("text {{placeholder}} text", "{{", "}}").collect();
        assert_eq!(
            tokens,
            vec![
                Token::Text("text "),
                Token::Placeholder("placeholder"),
                Token::Text(" text")
            ]
        );
    }

    #[test]
    fn test_multiple_chars_boundary_end_of_line() {
        let tokens: Vec<Token> = TokenIterator::new("text {{placeholder}}", "{{", "}}").collect();
        assert_eq!(
            tokens,
            vec![Token::Text("text "), Token::Placeholder("placeholder")]
        );
    }

    #[test]
    fn test_multiple_chars_boundary_multiple_boundaries() {
        let tokens: Vec<Token> = TokenIterator::new(
            "{{placeholder}} text {{placeholder}} test {{placeholder}}",
            "{{",
            "}}",
        )
        .collect();
        assert_eq!(
            tokens,
            vec![
                Token::Text(""),
                Token::Placeholder("placeholder"),
                Token::Text(" text "),
                Token::Placeholder("placeholder"),
                Token::Text(" test "),
                Token::Placeholder("placeholder")
            ]
        );
    }

    #[test]
    fn test_multiple_chars_missing_boundary_start() {
        let tokens: Vec<Token> = TokenIterator::new("text placeholder}}", "{{", "}}").collect();
        assert_eq!(tokens, vec![Token::Text("text placeholder}}")]);
    }

    #[test]
    fn test_multiple_chars_missing_boundary_end() {
        let tokens: Vec<Token> = TokenIterator::new("text {{placeholder", "{{", "}}").collect();
        assert_eq!(
            tokens,
            vec![Token::Text("text "), Token::Text("{{placeholder")]
        );
    }

    #[test]
    fn test_multiple_chars_boundary_and_missing_boundaries() {
        let tokens: Vec<Token> =
            TokenIterator::new("text {{placeholder}} {{placeholder", "{{", "}}").collect();
        assert_eq!(
            tokens,
            vec![
                Token::Text("text "),
                Token::Placeholder("placeholder"),
                Token::Text(" "),
                Token::Text("{{placeholder")
            ]
        );
    }

    #[test]
    fn test_trim_placeholdler_prefix_space() {
        let tokens: Vec<Token> = TokenIterator::new("text [ placeholder]", "[", "]").collect();
        assert_eq!(
            tokens,
            vec![Token::Text("text "), Token::Placeholder("placeholder")]
        );
    }

    #[test]
    fn test_trim_placeholdler_suffix_space() {
        let tokens: Vec<Token> = TokenIterator::new("text [placeholder ]", "[", "]").collect();
        assert_eq!(
            tokens,
            vec![Token::Text("text "), Token::Placeholder("placeholder")]
        );
    }

    #[test]
    fn test_trim_placeholdler_presuffix_space() {
        let tokens: Vec<Token> = TokenIterator::new("text [ placeholder ]", "[", "]").collect();
        assert_eq!(
            tokens,
            vec![Token::Text("text "), Token::Placeholder("placeholder")]
        );
    }
}
